"""Cell-level gene replacement in the SCimilarity encoder (Supplementary Fig. 1b, c).

Question: if a subtype's top-K genes (RECAST / COSG / Wilcoxon) are replaced in single target cells by the values of a
paired sibling cell (primary) or set to zero (sensitivity), do the cells lose their subtype identity in the encoder's
embedding?  Identity = k-nearest-neighbour majority label among the original embeddings of the lineage (self
excluded).

Inputs:
  units      subtype_benchmark/results/fs_uniq_perstate_v3.csv  (tier fine*, 61 units)
  cells      select_recast.load_lineage(lineage)                   (the cells, labels and gene order RECAST selected from)
  RECAST     subtype_benchmark/panels/topN_<lineage>_recast_M0.csv
  COSG       subtype_benchmark/fs_genesets_v2/<lineage>.csv  method=cosg
  Wilcoxon   subtype_benchmark/fs_genesets_v2/<lineage>.csv  method=scanpy_wilcoxontiecorrect_rankby_raw
  donors     program_scoring/donor_maps/<stem>_donor.tsv            (absent for hlca_lung / smillie_colon)
Env: SC (GPU).  Reads only."""
import os, sys, zlib
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F

SB = "/path/to/research/analysis"
FSB = f"{SB}/subtype_benchmark"
CD = f"{FSB}/cell_replacement"
ATLASES = f"{SB}/atlases"
for p in (SB, f"{SB}/ans_replication", FSB):
    if p not in sys.path: sys.path.insert(0, p)
import select_recast as FS                      # noqa: E402  load_lineage(): the benchmark's cells
import lineage_registry as C                      # noqa: E402
from cluster_key import pick                   # noqa: E402  make.names-tolerant cluster lookup
from scoring_benchmark.datasets import SCIMILARITY_MODEL  # noqa: E402
import recast                                     # noqa: E402
from recast.centroid import lognorm_rows          # noqa: E402
from recast.contrast import contrast_direction    # noqa: E402

METHODS = ["RECAST", "COSG", "Wilcoxon"]
DE_ID = {"COSG": "cosg", "Wilcoxon": "scanpy_wilcoxontiecorrect_rankby_raw"}
FIELD_ID = {"RECAST": "recast_M0", **DE_ID}         # method ids in fs_uniq_perstate_v3.csv
KS = [5, 10, 20, 50]
K_MAIN = 20
KNN = 15
CAP = 2000                                          # target / sibling-control cells per unit (seeded subsample)
N_NULL = 100
N_BINS = 10
DONOR_DIR = f"{SB}/program_scoring/donor_maps"


def seed_of(*parts):
    """Stable 32-bit seed from strings (crc32), so every draw is reproducible and independent of run order."""
    return zlib.crc32("|".join(map(str, parts)).encode())


# ---------------------------------------------------------------- cohort
def units61():
    """The 61 units: the fine tier of fs_uniq_perstate_v3.csv (subtypes with CellMarker-unique markers)."""
    u = pd.read_csv(f"{FSB}/results/fs_uniq_perstate_v3.csv")
    u = u[u["tier"].astype(str).str.startswith("fine")]
    pairs = sorted({(a, b) for a, b in zip(u.lineage, u.state)})
    assert len(pairs) == 61 and len({a for a, _ in pairs}) == 20, (len(pairs), len({a for a, _ in pairs}))
    return pairs


def gene_lists(lineage):
    """{method: {state: ranked genes}} for one lineage."""
    m0 = pd.read_csv(f"{FSB}/panels/topN_{lineage}_recast_M0.csv").sort_values(["cluster", "rank"])
    out = {"RECAST": {str(c): list(g.gene.astype(str)) for c, g in m0.groupby("cluster")}}
    de = pd.read_csv(f"{FSB}/fs_genesets_v2/{lineage}.csv").sort_values(["method", "cluster", "rank"])
    for name, mid in DE_ID.items():
        d = de[de.method == mid]
        out[name] = {str(c): list(g.gene.astype(str)) for c, g in d.groupby("cluster")}
    return out


def top_in_vocab(genes, gi, k):
    """First k genes of a ranked list that exist in the encoder vocabulary; (indices, n_skipped)."""
    idx, skipped = [], 0
    for g in genes:
        if g in gi:
            idx.append(gi[g])
            if len(idx) == k: break
        else:
            skipped += 1
    return np.asarray(idx, dtype=np.int64), skipped


# ---------------------------------------------------------------- lineage data
class Lineage:
    """Cells, labels, log-normalised input and donors of one lineage, exactly as load_lineage() returns them."""

    def __init__(self, lineage):
        self.name = lineage
        Craw, labels, keepstates, go = FS.load_lineage(lineage)
        self.labels = np.asarray(labels).astype(str)
        self.states = list(keepstates)
        self.go = list(go)
        self.gi = {g: i for i, g in enumerate(self.go)}
        self.Craw = Craw
        self.X = lognorm_rows(Craw).tocsr()                # per-cell log1p(1e4 x / sum x), SCimilarity's input
        # cell names: same rule as load_lineage (labels file / min_cells), read in backed mode; labels must agree
        e = C.LINEAGES[lineage]
        self.stem = e["ds"]
        lab_all = np.array([l.strip() for l in open(f"{ATLASES}/{e['label_file']}")])
        a = ad.read_h5ad(f"{ATLASES}/{self.stem}_attribution_scimilarity.h5ad", backed="r")
        obs = np.asarray(a.obs_names).astype(str); a.file.close()
        keep = np.isin(lab_all, keepstates)
        assert keep.sum() == len(self.labels) and (lab_all[keep] == self.labels).all(), f"{lineage}: cell set drift"
        self.cells = obs[keep]
        p = f"{DONOR_DIR}/{self.stem}_donor.tsv"
        if os.path.exists(p):
            dm = pd.read_csv(p, sep="\t").set_index("cell")["donor"].astype(str)
            self.donor = dm.reindex(self.cells).to_numpy(dtype=object)
        else:
            self.donor = np.full(len(self.cells), None, dtype=object)
        self.n = len(self.labels)

    def rows(self, idx, cols):
        """Dense float32 [len(idx), len(cols)] block of the log-normalised input."""
        return self.X[idx][:, cols].toarray().astype(np.float32)


# ---------------------------------------------------------------- encoder
class Enc:
    """SCimilarity network split at its first Linear, so a K-gene edit is a rank-K update of that layer's output."""

    def __init__(self, device="cuda"):
        self.enc = recast.SCimilarityEncoder(SCIMILARITY_MODEL, device=device, normalize=False)
        self.net = self.enc._net
        self.net.eval()
        assert not self.net.training
        self.dev = next(self.net.parameters()).device
        blk0 = list(self.net.network[0])
        lin_pos = [i for i, m in enumerate(blk0) if isinstance(m, nn.Linear)]
        assert len(lin_pos) == 1, blk0
        self.pre = blk0[:lin_pos[0]]                    # Dropout only: identity in eval mode
        assert all(isinstance(m, nn.Dropout) for m in self.pre), self.pre
        self.lin0 = blk0[lin_pos[0]]
        self.post = blk0[lin_pos[0] + 1:]
        self.rest = list(self.net.network[1:])
        self.W = self.lin0.weight.detach()              # [1024, G]

    @torch.inference_mode()
    def H0(self, X, bs=4096):
        """First-Linear output for every row of a (sparse) input matrix, on the GPU."""
        parts = []
        for i in range(0, X.shape[0], bs):
            B = X[i:i + bs]
            B = B.toarray() if sp.issparse(B) else np.asarray(B)
            parts.append(self.lin0(torch.as_tensor(B, dtype=torch.float32, device=self.dev)))
        return torch.cat(parts)

    @torch.inference_mode()
    def head(self, H):
        for m in self.post: H = m(H)
        for m in self.rest: H = m(H)
        return F.normalize(H, p=2, dim=1)

    @torch.inference_mode()
    def edited(self, H0, dX, cols):
        """Embeddings after adding dX [n, K] to input columns `cols` (rank-K update of the first Linear)."""
        if len(cols) == 0:
            return self.head(H0.clone())
        d = torch.as_tensor(dX, dtype=torch.float32, device=self.dev)
        return self.head(H0 + d @ self.W[:, torch.as_tensor(cols, device=self.dev)].T)

    @torch.inference_mode()
    def full(self, Xdense):
        """Reference path: the whole network on dense input (used only by the gates)."""
        return self.net(torch.as_tensor(Xdense, dtype=torch.float32, device=self.dev))


# ---------------------------------------------------------------- k-nearest-neighbour identity
class KNN:
    """Majority label among the k nearest ORIGINAL embeddings (cosine; embeddings are unit-norm), self excluded.
    Ties are broken by the summed similarity of the tied classes."""

    def __init__(self, Zref, labels, states, k=KNN):
        self.R = Zref                                   # [n, d] torch, unit rows
        self.code = {s: i for i, s in enumerate(states)}
        lab = torch.as_tensor([self.code[s] for s in labels], device=Zref.device)
        self.onehot = F.one_hot(lab, len(states)).float()   # [n, S]
        self.k = k

    @torch.inference_mode()
    def predict(self, Q, excl, bs=2048):
        """Predicted state index for each query row.  excl = list of index arrays (same length as Q); for query i
        every excl[j][i] reference row is removed from its neighbour set (the cell itself, and for a replaced cell
        also the sibling partner whose values it received, so a vote cannot come from the copied cell)."""
        out = []
        ex = [torch.as_tensor(np.asarray(e), device=Q.device) for e in excl]
        for i in range(0, Q.shape[0], bs):
            S = Q[i:i + bs] @ self.R.T
            r = torch.arange(S.shape[0], device=Q.device)
            for e in ex:
                S[r, e[i:i + bs]] = -float("inf")
            v, j = S.topk(self.k, dim=1)
            oh = self.onehot[j].double()                 # [b, k, S]
            # exact lexicographic rule (design section 5): most votes first, then the larger summed similarity.
            # Summed similarity is bounded by k, so votes * (k + 1) keeps counts strictly dominant; float64 keeps
            # sub-1e-3 similarity differences that a float32 score would collapse (ties would then fall to the
            # alphabetically first state).
            score = oh.sum(1) * (self.k + 1) + (oh * v.double().unsqueeze(-1)).sum(1)
            out.append(score.argmax(1))
        return torch.cat(out).cpu().numpy()


# ---------------------------------------------------------------- partners and null genes
def sibling_partners(L, tgt_idx, sib_idx, unit):
    """One sibling cell per target cell: same donor when the dataset has a donor map and that donor has sibling
    cells, otherwise any sibling.  Returns (partner indices, same-donor flags)."""
    rng = np.random.default_rng(seed_of("partner", unit))
    ok = lambda d: d is not None and d == d          # None / NaN = no donor information
    by_donor = {}
    for i in sib_idx:
        if ok(L.donor[i]):
            by_donor.setdefault(L.donor[i], []).append(i)
    part = np.empty(len(tgt_idx), dtype=np.int64); same = np.zeros(len(tgt_idx), dtype=bool)
    for n, t in enumerate(tgt_idx):
        pool = by_donor.get(L.donor[t]) if ok(L.donor[t]) else None
        if pool:
            part[n] = pool[rng.integers(len(pool))]; same[n] = True
        else:
            part[n] = sib_idx[rng.integers(len(sib_idx))]
    return part, same


def null_gene_sets(L, tgt_idx, method_cols, exclude, unit, method, n=N_NULL, n_bins=N_BINS):
    """n expression-matched random gene sets for one method's genes: genes binned by mean input value in the target
    cells (10 equal-frequency bins over expressed genes + one zero bin); each method gene is replaced by a gene from
    its own bin, drawn without replacement within a set, never from `exclude`."""
    mu = np.asarray(L.X[tgt_idx].mean(axis=0)).ravel()
    expressed = mu > 0
    edges = np.quantile(mu[expressed], np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_of = np.where(expressed, 1 + np.searchsorted(edges, mu, side="right"), 0)
    allowed = np.ones(len(mu), dtype=bool); allowed[list(exclude)] = False
    pools = {b: np.flatnonzero((bin_of == b) & allowed) for b in np.unique(bin_of)}
    rng = np.random.default_rng(seed_of("null", unit, method))
    sets = []
    for _ in range(n):
        chosen = []
        for c in method_cols:
            pool = pools[bin_of[c]]
            pool = pool[~np.isin(pool, chosen)]
            chosen.append(int(pool[rng.integers(len(pool))]))
        sets.append(np.asarray(chosen, dtype=np.int64))
    return sets


# ---------------------------------------------------------------- read-outs
def readouts(Zq0, Zq1, pred1, own_code, anchored, u, dist):
    """identity loss among anchored cells; signed shift toward the reference along u (1 = full centroid distance);
    orthogonal displacement relative to the same distance."""
    D = (Zq1 - Zq0)
    along = -(D @ u)                                     # u points reference -> target; positive = toward reference
    orth = torch.linalg.norm(D + along.unsqueeze(1) * u.unsqueeze(0), dim=1)
    a = anchored
    return dict(
        n_anchored=int(a.sum()),
        identity_loss=float((pred1[a] != own_code).mean()) if a.any() else np.nan,
        axis_shift=float(along.mean().item() / dist),
        orth_shift=float(orth.mean().item() / dist),
    )
