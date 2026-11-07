#!/usr/bin/env python
"""scMarkerGene (Brief Bioinform 2026) on one subtype lineage, end to end.

Stages, in the order of the original code:
  001  prepare data          vectorised version, identical output to the original on the original demo data
                             (see scmarkergene/)
  002  train + explain       the original code, except that DeepLIFT runs batched and writes float32 npy instead of
                             ~700 MB of text per model; batching was checked against the batch-of-one loop on identical
                             weights
  003  extract markers       the original statistics and selection functions, called through a vectorised data layer

Depth: the method's defaults are 20 markers per cell type out of a 50-gene candidate pool.  The benchmark reads
rankings at 20 and 200, so both are raised to 200 and the depth the method's own filters reach is recorded (never
padded).  Ranking within a cell type is the order `select_marker_genes` emits; its `marker_score` is kept.

GPU/CPU split: 002 is the only GPU stage.  003's permutation FDR is CPU-heavy and can run on the CPU server with
`--stage aggregate` (writes a compact bundle) and then `--stage select`.

->  genesets200/<lin>__scmarkergene.csv   method,cluster,rank,gene
->  timing/<lin>__scmarkergene.json       per-stage wall time, plus the depth reached"""
import argparse, gzip, json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = f"{HERE}/scmarkergene"
SRC = f"{HERE}/scmarkergene_src"


def log(msg):
    print(f"[scMarkerGene] {msg}", flush=True)


def claim(work_dir, lineage):
    """Take an exclusive claim on this lineage, or return False if another worker holds it.

    Several drivers run over the 20 lineages at once and every stage of this pipeline writes into
    one shared `scmg_work/<lin>/` tree, so two workers on the same lineage would overwrite each
    other's optuna store, checkpoints and attribution arrays.  A driver's "output already exists"
    test cannot see a run that is still in flight, so the claim lives here instead: an O_EXCL file
    is atomic, and a claim whose pid is gone is stale and gets taken over.
    """
    lock = f"{work_dir}/.claim"
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(open(lock).read().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            alive = pid > 0
            if alive:
                try:
                    os.kill(pid, 0)
                except OSError:
                    alive = False
            if alive:
                log(f"{lineage}: claimed by pid {pid}, skipping")
                return False
            log(f"{lineage}: stale claim from pid {pid or '?'}, taking it over")
            try:
                os.unlink(lock)
            except FileNotFoundError:
                pass


def release(work_dir):
    try:
        if int(open(f"{work_dir}/.claim").read().strip()) == os.getpid():
            os.unlink(f"{work_dir}/.claim")
    except (OSError, ValueError):
        pass


def run(cmd, env=None, log_path=None):
    e = dict(os.environ)
    e.update(env or {})
    if log_path:
        with open(log_path, "a") as fo:
            r = subprocess.run(cmd, env=e, stdout=fo, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, env=e)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd)}  -- see {log_path}")


def write_dense_tsv(sce, lin, good_genes, out_tsv, chunk=512):
    """`extractMarkerGenes` reads a dense TSV of RAW counts: cell_id, group, one column per gene.

    Only the genes 001 kept are written -- candidates are always a subset of those, and the full
    matrix would be several GB of text per lineage.
    """
    import scipy.io as sio, scipy.sparse as sp
    with gzip.open(f"{sce}/{lin}_counts.mtx.gz", "rb") as fh:
        m = sio.mmread(fh)
    X = sp.csr_matrix(m.T)
    genes = np.array([l.strip() for l in open(f"{sce}/{lin}_genes.txt")], dtype=object)
    labels = [l.strip() for l in open(f"{sce}/{lin}_labels.txt")]
    idx = {g: i for i, g in enumerate(genes.astype(str))}
    cols = np.array([idx[g] for g in good_genes], dtype=int)
    with open(out_tsv, "w") as fo:
        fo.write("cell_id\tgroup\t" + "\t".join(good_genes) + "\n")
        for s in range(0, X.shape[0], chunk):
            blk = np.asarray(X[s:s + chunk][:, cols].todense())
            for r in range(blk.shape[0]):
                i = s + r
                fo.write(f"cell{i}\t{labels[i]}\t"
                         + "\t".join("%g" % v for v in blk[r]) + "\n")


def read_markers(path):
    """scMarkerGene's Top-N table -> {cell type: [gene, ...] in the emitted order}."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t")
    out = {}
    for ct, sub in df.groupby("target_cell_type", sort=False):
        out[str(ct)] = [str(g) for g in sub["marker"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", required=True)
    ap.add_argument("--sce", default=f"{HERE}/../sce_export_v2")
    ap.add_argument("--work", default=f"{HERE}/scmg_work")
    ap.add_argument("--out", default=f"{HERE}/genesets200")
    ap.add_argument("--timing", default=f"{HERE}/timing")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--topn", type=int, default=200)
    ap.add_argument("--marker-num", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strategy", default="upsample", choices=["upsample", "weights", "none"])
    ap.add_argument("--keep-attributions", action="store_true",
                    help="keep the per-model npy files (tens of GB per lineage)")
    ap.add_argument("--stage", default="all", choices=["all", "prep", "train", "select"])
    a = ap.parse_args()

    lin = a.lineage
    W = f"{a.work}/{lin}"
    os.makedirs(W, exist_ok=True)
    if not claim(W, lin):
        return
    import atexit
    atexit.register(release, W)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.timing, exist_ok=True)
    prep = f"{W}/001_prepareData_output"
    gc_out = f"{W}/002_geneContribution_output"
    logf = f"{W}/run.log"
    times = {}
    env = {"PYTHONPATH": f"{HERE}/pylibs", "NUMPY_MADVISE_HUGEPAGE": "0"}

    if a.stage in ("all", "prep"):
        log(f"{lin}: 001 prepare")
        t = time.perf_counter()
        run([a.python, f"{PORT}/001_prepareData_fast.py",
             "--mtx", f"{a.sce}/{lin}_counts.mtx.gz",
             "--genes", f"{a.sce}/{lin}_genes.txt",
             "--labels", f"{a.sce}/{lin}_labels.txt",
             "-o", prep, "-t", "count", "--seed", str(a.seed)], env, logf)
        times["prep_s"] = time.perf_counter() - t
        good = [l.strip() for l in open(f"{prep}/orderedGeneList.dat")]
        log(f"{lin}: dense TSV for the marker stage ({len(good)} genes)")
        t = time.perf_counter()
        write_dense_tsv(a.sce, lin, good, f"{W}/expr.tsv")
        times["tsv_s"] = time.perf_counter() - t

    if a.stage in ("all", "train"):
        log(f"{lin}: 002 train + DeepLIFT")
        t = time.perf_counter()
        run([a.python, f"{PORT}/002_geneContribution_fast.py",
             "-i", prep, "-o", gc_out, "--strategy", a.strategy,
             "--optuna_storage", f"sqlite:///{os.path.abspath(W)}/optuna.sqlite3"], env, logf)
        times["train_s"] = time.perf_counter() - t

    if a.stage in ("all", "select"):
        log(f"{lin}: 003 aggregate + select")
        t = time.perf_counter()
        run([a.python, f"{PORT}/003_extractCellMarker_fast.py",
             "-i", W, "-o", f"{W}/003_extractCellMarker_output", "-e", f"{W}/expr.tsv",
             "-p", str(a.n_perm), "-n", str(a.marker_num), "-t", str(a.topn)], env, logf)
        times["select_s"] = time.perf_counter() - t

        mk = read_markers(f"{W}/003_extractCellMarker_output/"
                          f"scMarkerGene_Top{a.marker_num}_marker.txt")
        import pandas as pd
        rows = [("scmarkergene", ct, i + 1, g) for ct, gl in mk.items() for i, g in enumerate(gl)]
        pd.DataFrame(rows, columns=["method", "cluster", "rank", "gene"]).to_csv(
            f"{a.out}/{lin}__scmarkergene.csv", index=False)
        depths = {ct: len(gl) for ct, gl in mk.items()}
        times.update({"lineage": lin, "method": "scmarkergene",
                      "seconds": sum(v for k, v in times.items() if k.endswith("_s")),
                      "n_clusters": len(mk), "min_depth": min(depths.values(), default=0),
                      "max_depth": max(depths.values(), default=0), "depth_per_cluster": depths,
                      "requested_depth": a.marker_num, "n_perm": a.n_perm,
                      "note": "wall time includes GPU training; not comparable to the CPU field"})
        json.dump(times, open(f"{a.timing}/{lin}__scmarkergene.json", "w"), indent=1)
        log(f"{lin}: done, depth {times['min_depth']}-{times['max_depth']} over {len(mk)} subtypes")

        if not a.keep_attributions:
            for f in os.listdir(f"{gc_out}/explain_dir"):
                if f.endswith(".npy"):
                    os.remove(f"{gc_out}/explain_dir/{f}")
            if os.path.exists(f"{W}/expr.tsv"):
                os.remove(f"{W}/expr.tsv")


if __name__ == "__main__":
    main()
