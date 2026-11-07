"""Donor-clustered bootstrap intervals for the per-cell scoring differences of Fig. 6a.

Cells within a donor are correlated, so donors are resampled with replacement inside each task and both arms are
evaluated again on the drawn cells.

Donor sources:
  baron_pancreas / pelka_crc_epi   the *_raw.h5ad next to the task: obs['donor'] / obs['specimen'] (positional join,
                                   label agreement checked == 1.0)
  5 atlases                        donor_maps/<parent>_donor.tsv, exported from the atlas objects on the CPU server
                                   (donor_id / sample), joined by cell name (all cells found)
  everything else                  no donor unit -> no interval
Tasks with fewer than MIN_DONORS donors get no interval either: Baron has 4 donors and Pelka 2 specimens.

The point estimates are recomputed here and checked against reuse_by_task.csv."""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, anndata as ad

SB = "/path/to/research/analysis"
sys.path.insert(0, f"{SB}/ans_replication")
sys.path.insert(0, f"{SB}/attribution_methods")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_programs as B                      # noqa: E402
import ans_baselines as ANSB                       # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

D6 = f"{SB}/paper/program_scoring_tables/data"
DM = f"{SB}/program_scoring/donor_maps"
OUT = f"{SB}/program_scoring/results_donorci"
MIN_DONORS = 5
RAW = {"baron_pancreas": (f"{SB}/dataset_files/baron_pancreas/baron_pancreas_raw.h5ad", "donor"),
       "pelka_crc_epi":  (f"{SB}/dataset_files/pelka_crc_epi/pelka_crc_epi_raw.h5ad", "specimen")}


def selection(name):
    """Mirror load_dataset's keep-mask + stratified cap, returning the surviving row indices."""
    cfg = B.DATASETS[name]
    a = ad.read_h5ad(cfg["h5ad"], backed="r")
    y = np.array([l.strip() for l in open(cfg["labels"])])
    assert len(y) == a.n_obs, name
    lmap = cfg.get("label_map", {})
    if lmap:
        y = np.array([lmap.get(v, v) for v in y])
    panels = json.load(open(cfg["sig_json"])) if cfg["regime"] == "distinct" \
        else {s: cfg["panels"][s] for s in cfg["states"] if s in cfg["panels"]}
    cand = sorted(c for c in panels if int((y == c).sum()) >= B.MIN_CELLS)
    idx = np.where(np.isin(y, cand))[0]
    yk = y[idx]
    if idx.size > B.CAP:
        rng = np.random.RandomState(0)
        per = max(B.MIN_CELLS, B.CAP // len(cand))
        sel = [np.where(yk == c)[0] if (yk == c).sum() <= per
               else rng.choice(np.where(yk == c)[0], per, replace=False) for c in cand]
        idx = idx[np.sort(np.concatenate(sel))]
    return idx, np.array(a.obs_names, dtype=object)[idx], cand


def donors_for(name, parent, idx, cells):
    if name in RAW:
        rp, col = RAW[name]
        r = ad.read_h5ad(rp, backed="r")
        return np.array(r.obs[col], dtype=object)[idx], "raw:" + col
    f = os.path.join(DM, f"{parent}_donor.tsv")
    if os.path.exists(f):
        d = pd.read_csv(f, sep="\t")
        m = dict(zip(d.cell.astype(str), d.donor.astype(str)))
        got = np.array([m.get(str(c)) for c in cells], dtype=object)
        assert all(g is not None for g in got), f"{name}: donor join incomplete"
        return got, "tsv"
    return None, "none"


def arms(name):
    """RECAST z_train out-of-fold, ANS, and SIGnature (gated) score matrices on the same cells."""
    X, y, cand, sig_idx, sig_genes, regime = B.load_dataset(name)
    n, K = X.shape[0], len(cand)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=B.CV_SEED)
    S = {}
    oof = np.zeros((n, K), "float32")
    for tr, te in skf.split(X, y):
        p = B.build_candidate_params(X[tr], y[tr], cand, sig_idx)
        S_tr, S_te = B.m1_scores(X[tr], p, cand, sig_idx), B.m1_scores(X[te], p, cand, sig_idx)
        mu, sd = S_tr.mean(0, keepdims=True), S_tr.std(0, keepdims=True); sd[sd == 0] = 1.0
        oof[te] = (S_te - mu) / sd
    S["recast"] = oof
    A = ad.AnnData(X=X, var=pd.DataFrame(index=B.GO)); A.obs_names = [str(i) for i in range(n)]
    # As score_all_methods: a candidate whose curated panel keeps no gene in the SCimilarity space scores NaN,
    # which the argmax never picks.
    _ans = []
    for c in cand:
        try:
            _ans.append(ANSB.score_cells("ANS", A, sig_genes[c]))
        except Exception as e:                               # noqa: BLE001
            print(f"  [ANS NaN] {name} x {c}: {type(e).__name__}: {e}", flush=True)
            _ans.append(np.full(n, np.nan, dtype="float64"))
    S["ans"] = np.column_stack(_ans).astype("float32")
    try:
        import reference_free_scoring as SIG
        Xs, Attr, ys, cands, sidx, _sgen, _r = SIG.load_dataset_with_attr(name)
        assert (ys == y).all() and cands == cand, f"{name}: SIG loader returned different cells"
        so = np.zeros((n, K), "float32")
        for tr, te in skf.split(Xs, ys):
            par = SIG.reference_free_params(Attr[tr], Xs[tr], ys[tr], cand)
            so[te] = SIG._ztrain(SIG.reference_free_scores(Xs[tr], par, cand, sidx, gated=True),
                                 SIG.reference_free_scores(Xs[te], par, cand, sidx, gated=True))
        S["sig"] = so
    except Exception as e:                                   # noqa: BLE001
        print(f"  [WARN] SIG arm unavailable for {name}: {type(e).__name__}: {e}", flush=True)
    return S, y, cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    master = pd.read_csv(f"{D6}/reuse_by_task.csv").set_index("dataset")
    sigpub = pd.read_csv(f"{D6}/reuse_by_task_reference_free.csv").set_index("dataset")
    names = list(master.index) if A.dataset == "all" else [A.dataset]
    rows = []
    tag = "" if A.dataset == "all" else f"_{A.dataset}"
    dest = os.path.join(OUT, f"reuse_donor_intervals{tag}.csv")
    for name in names:
        t0 = time.time()
        idx, cells, cand0 = selection(name)
        don, src = donors_for(name, master.loc[name, "parent"], idx, cells)
        nd = 0 if don is None else len(set(don))
        S, y, cand = arms(name)
        assert cand == cand0, f"{name}: candidate mismatch between selection and loader"
        # check: the point estimates equal reuse_by_task.csv
        pt = {k: B.evaluate(k, v, y, cand) for k, v in S.items()}
        for k, col in (("recast", "recast"), ("ans", "ans")):
            for mk, mc in (("balanced_acc", "ba"), ("macro_f1", "f1"), ("mean_auroc", "auc")):
                pub = master.loc[name, f"{col}_{mc}"]
                assert abs(pt[k][mk] - pub) < 1e-6, f"{name} {k} {mc}: {pt[k][mk]} vs table {pub}"
        if "sig" in pt and name in sigpub.index:      # same check for the SIGnature arm
            for mk, mc in (("balanced_acc", "ba"), ("macro_f1", "f1"), ("mean_auroc", "auc")):
                pub = sigpub.loc[name, f"sig_gated_{mc}"]
                assert abs(pt["sig"][mk] - pub) < 1e-6, f"{name} sig {mc}: {pt['sig'][mk]} vs table {pub}"
        base = {c: {m: pt[c][m] for m in ("balanced_acc", "macro_f1", "mean_auroc")} for c in pt}
        if don is None or nd < MIN_DONORS:
            for comp in [c for c in ("ans", "sig") if c in S]:
                for mk, mc in (("balanced_acc", "ba"), ("macro_f1", "f1"), ("mean_auroc", "auc")):
                    rows.append(dict(dataset=name, comparator=comp, metric=mc,
                                     delta=base["recast"][mk] - base[comp][mk],
                                     ci_lo=np.nan, ci_hi=np.nan, n_donors=nd, donor_src=src,
                                     n_boot=0, degenerate=0))
            pd.DataFrame(rows).to_csv(dest, index=False)          # flush: never lose finished tasks
            print(f"[{name}] donors={nd} ({src}) -> NO CI  ({time.time()-t0:.0f}s)", flush=True)
            continue
        uniq = np.array(sorted(set(don)))
        by = {d: np.where(don == d)[0] for d in uniq}
        rng = np.random.default_rng(A.seed)
        acc = {c: {m: [] for m in ("balanced_acc", "macro_f1", "mean_auroc")} for c in S if c != "recast"}
        degen = 0
        for _ in range(A.B):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            bi = np.concatenate([by[d] for d in pick])
            yb = y[bi]
            if len(set(yb)) < len(cand):
                degen += 1
            ev = {c: B.evaluate(c, S[c][bi], yb, cand) for c in S}
            for c in acc:
                for m in acc[c]:
                    acc[c][m].append(ev["recast"][m] - ev[c][m])
        for comp in acc:
            for mk, mc in (("balanced_acc", "ba"), ("macro_f1", "f1"), ("mean_auroc", "auc")):
                v = np.array(acc[comp][mk], float); v = v[np.isfinite(v)]
                rows.append(dict(dataset=name, comparator=comp, metric=mc,
                                 delta=base["recast"][mk] - base[comp][mk],
                                 ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5),
                                 n_donors=nd, donor_src=src, n_boot=len(v), degenerate=degen))
        pd.DataFrame(rows).to_csv(dest, index=False)              # flush: never lose finished tasks
        d_ba = [r for r in rows if r["dataset"] == name and r["comparator"] == "ans" and r["metric"] == "ba"][0]
        print(f"[{name}] donors={nd} ({src}) B={A.B} degen={degen}  "
              f"ANS ba delta={d_ba['delta']:+.4f} [{d_ba['ci_lo']:+.4f},{d_ba['ci_hi']:+.4f}]  "
              f"({time.time()-t0:.0f}s)", flush=True)
    print(f"\n[written] {OUT}/reuse_donor_intervals{tag}.csv  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
