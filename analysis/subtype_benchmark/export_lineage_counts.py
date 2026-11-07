"""Export the integer counts of every subtype lineage for the comparator methods (sce_export_v2/).

Counts come from <stem>_fullgene.h5ad (checked integer) and are aligned to the 28,231-gene SCimilarity space with
align_dataset; the counts layer of the attribution h5ad files is log-normalized (SIGnature's lognorm_counts writes X
into layers['counts'] in place) and is not used.  Cells, subtypes, gene filter (genes detected in >= 3 cells) and file
layout follow lineage_registry and extra_lineages.

Env: SC.  Usage: python export_lineage_counts.py [dsid ...]   (default: all)"""
import gzip
import os
import shutil
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/path/to/research/external/Signature/src")
import data as sig_data  # noqa: E402
import lineage_registry as FC  # noqa: E402
from extra_lineages import LINEAGES as EXTRA  # noqa: E402

MODEL = "/path/to/scimilarity/model_v1.1"
SB = "/path/to/research/analysis"
ATLASES = f"{SB}/atlases"
OUT = f"{SB}/subtype_benchmark/sce_export_v2"
GO = sig_data.read_gene_order(MODEL)
MIN_STATE = 60

IMMUNE = {
    "fs_pbmcbmn_l2": ("pbmcbmn", "bench_labels_pbmcbmn_l2.txt",
                      ["B naive", "B intermediate", "B memory"]),
    "fs_yoshida": ("yoshida", "bench_labels_yoshida.txt",
                   ["Naive B cells", "Non-switched memory B cells", "Switched memory B cells",
                    "Memory B cells"]),
    "fs_helperT": ("dominguez_conde", "bench_labels_dominguez_conde.txt",
                   ["Tcm/Naive helper T cells", "Tem/Effector helper T cells",
                    "Regulatory T cells"]),
    "fs_cytoT": ("dominguez_conde", "bench_labels_dominguez_conde.txt",
                 ["Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells",
                  "Tem/Trm cytotoxic T cells", "gdT"]),
}
RADC_PARENT = {"fs_radc_in": "IN", "fs_radc_en": "EN"}

_CACHE = {}


def real_counts(stem):
    """cells x 28,231 integer counts, rows in the attribution h5ad's cell order."""
    if stem in _CACHE:
        return _CACHE[stem]
    from SIGnature.utils import align_dataset

    A = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad", backed="r")
    want = list(map(str, A.obs_names))
    A.file.close()

    fg = ad.read_h5ad(f"{ATLASES}/{stem}_fullgene.h5ad")
    pos = {str(b): i for i, b in enumerate(fg.obs_names)}
    missing = [b for b in want if b not in pos]
    if missing:
        raise RuntimeError(f"{stem}: {len(missing)} barcodes absent from fullgene")
    X = sp.csr_matrix(fg.X)[[pos[b] for b in want]]
    raw = ad.AnnData(X=X, var=pd.DataFrame(index=fg.var_names))
    del fg
    raw.var_names_make_unique()
    aligned = align_dataset(raw, GO)
    C = sp.csr_matrix(aligned.X)
    C.eliminate_zeros()
    d = C.data
    if not (C.nnz and np.all(d == np.round(d))):
        raise RuntimeError(f"{stem}: aligned matrix is not integer counts")
    print(f"  [{stem}] real counts {C.shape}, max={d.max():.0f}, "
          f"median per-cell total={np.median(np.asarray(C.sum(1)).ravel()):.0f}", flush=True)
    _CACHE.clear()          # one stem in memory at a time (RADC is 121k x 28k)
    _CACHE[stem] = C
    return C


def write_ds(dsid, C, labels):
    """Gene filter + validation + write, mirroring the originals byte-for-byte in structure."""
    det = np.asarray((C > 0).sum(0)).ravel()
    gkeep = det >= 3
    Cg = C[:, gkeep]
    g = np.asarray(GO)[gkeep]

    Cg = sp.csr_matrix(Cg)
    Cg.eliminate_zeros()
    d = Cg.data
    assert np.all(d == np.round(d)), f"{dsid}: non-integer values"
    assert np.all(d > 0), f"{dsid}: explicit zeros survived"
    tot = np.median(np.asarray(Cg.sum(1)).ravel())
    assert tot > 200, f"{dsid}: implausible median per-cell total {tot}"

    os.makedirs(OUT, exist_ok=True)
    mtx = f"{OUT}/{dsid}_counts.mtx"
    sio.mmwrite(mtx, Cg.T.tocoo().astype(np.int32), field="integer")   # features x cells
    with open(mtx, "rb") as fi, gzip.open(mtx + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(mtx)
    pd.Series(g).to_csv(f"{OUT}/{dsid}_genes.txt", index=False, header=False)
    pd.Series(labels).to_csv(f"{OUT}/{dsid}_labels.txt", index=False, header=False)
    print(f"[{dsid}] {Cg.shape[0]} cells x {Cg.shape[1]} genes | max={d.max():.0f} "
          f"median_total={tot:.0f} | states={sorted(pd.unique(labels))}", flush=True)
    return dict(dsid=dsid, n_cells=int(Cg.shape[0]), n_genes=int(Cg.shape[1]),
                max_count=int(d.max()), median_cell_total=float(tot))


def jobs():
    """dsid -> (stem, callable(C) -> (Csub, labels)); grouped so each stem loads once."""
    out = {}
    for dsid, (stem, lf, states) in IMMUNE.items():
        out[dsid] = (stem, ("labelfile", lf, states, MIN_STATE, None))
    for dsid, parent in RADC_PARENT.items():
        out[dsid] = ("RADC", ("radc", parent, None, MIN_STATE, None))
    for tag, (ds, states) in EXTRA.items():
        out[f"ext_{tag}"] = (ds, ("obslabel", None, states, MIN_STATE, None))
    for dsid in FC.NEW_IDS:
        e = FC.LINEAGES[dsid]
        out[dsid] = (e["ds"], ("labelfile", e["label_file"], e["states"], e["min_cells"], None))
    return out


def subset(kind, C, stem, lf, states, minc):
    if kind == "labelfile":
        lab = np.array([l.strip() for l in open(f"{ATLASES}/{lf}")])
        assert len(lab) == C.shape[0], f"labels {len(lab)} != cells {C.shape[0]}"
        keep_states = sorted([s for s in states if (lab == s).sum() >= minc])
        keep = np.isin(lab, keep_states)
        return C[keep], lab[keep]
    if kind == "obslabel":
        A = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad", backed="r")
        lab = A.obs["label"].astype(str).to_numpy()
        A.file.close()
        keep_states = [s for s in states if (lab == s).sum() >= minc]
        keep = np.isin(lab, keep_states)
        return C[keep], lab[keep]
    if kind == "radc":
        cls = np.array([l.strip() for l in open(f"{ATLASES}/bench_labels_RADC_class.txt")])
        st = np.array([l.strip() for l in open(f"{ATLASES}/bench_labels_RADC_subclass.txt")])
        inlin = cls == lf                                   # lf carries the parent class here
        st_states = [s for s in pd.unique(st[inlin]) if (st[inlin] == s).sum() >= minc]
        keep = inlin & np.isin(st, st_states)
        return C[keep], st[keep]
    raise ValueError(kind)


def main():
    J = jobs()
    want = sys.argv[1:] or list(J)
    by_stem = {}
    for dsid in want:
        by_stem.setdefault(J[dsid][0], []).append(dsid)

    rows, failed = [], []
    for stem in sorted(by_stem):
        print(f"\n=== stem {stem} ({len(by_stem[stem])} datasets) ===", flush=True)
        try:
            C = real_counts(stem)
        except Exception as e:
            print(f"  !! {stem}: {type(e).__name__}: {e}", flush=True)
            failed += [(d, str(e)) for d in by_stem[stem]]
            continue
        for dsid in by_stem[stem]:
            _, (kind, lf, states, minc, _) = J[dsid]
            try:
                Csub, lab = subset(kind, C, stem, lf, states, minc)
                rows.append(write_ds(dsid, Csub, lab))
            except Exception as e:
                import traceback
                traceback.print_exc()
                failed.append((dsid, f"{type(e).__name__}: {e}"))

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(f"{OUT}/rebuild_summary.csv", index=False)
        print("\n=== REBUILT ===")
        print(df.to_string(index=False))
    if failed:
        print("\n=== FAILED ===")
        for d, e in failed:
            print(f"  {d}: {e}")
    print(f"\n{len(rows)} ok, {len(failed)} failed -> {OUT}")


if __name__ == "__main__":
    main()
