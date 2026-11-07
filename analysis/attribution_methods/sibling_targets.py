"""Attribution targets for one subtype against its sibling subtypes.  All four share the encoder, the
log-normalized input, the pooled-counts profiles, complete integrated gradients (multiply_by_inputs=True), the
positive-channel mask, the candidate genes and the panel size; only the target (and, for M3, the input and baseline)
differs.

  M0  direction  F = z.u_c,  u_c = normalize(mu_c - mu_sib)              (RECAST)
  M1  softmax    F = (z.mu_c)/tau - logsumexp_k((z.mu_k)/tau), k in sib(c)
  M2  orthogonal F = z.u_perp,  u_perp = normalize(v_sub - (v_sub.u_par) u_par)   (parent-lineage direction removed)
  M3  triplet    F = z.zp - z.zn, per target cell, zn = the nearest cell of the closest sibling, baseline = that
                 cell's expression

mu_* are per-subtype mean embeddings; for M1 each centroid is unit-normalized.  Uses population_ig's profiles and
contrastive_ig's encoder.  Every function returns signed integrated gradients on SCimilarity's gene order [G]."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/path/to/research/analysis/attribution_methods"); sys.path.insert(0, "/path/to/research/external/Signature/src")
import population_ig as L, contrastive_ig
from captum.attr import IntegratedGradients
enc, dev, embed, go, gi = contrastive_ig.enc, contrastive_ig.dev, contrastive_ig.embed, contrastive_ig.go, contrastive_ig.gi
ig_of = {i: g for g, i in gi.items()}
CentroidTarget = contrastive_ig.CentroidTarget                     # M0/M2: F = <enc(x), u>


class SoftmaxTarget(nn.Module):
    """M1: F(x) = (z.mu_c)/tau - logsumexp_{k in sib}( (z.mu_k)/tau ). muc/Msib are buffers (NOT captum
    additional args) so captum's per-step batch expansion never tiles the fixed [S,128] sibling matrix."""
    def __init__(self, encoder, muc, Msib, tau):
        super().__init__(); self.encoder = encoder; self.tau = float(tau)
        self.register_buffer("muc", torch.as_tensor(muc, dtype=torch.float32))     # [128]
        self.register_buffer("Msib", torch.as_tensor(Msib, dtype=torch.float32))   # [S,128]
    def forward(self, x):
        z = self.encoder(x)
        sc = (z * self.muc).sum(1) / self.tau            # [B]
        ss = (z @ self.Msib.T) / self.tau                # [B,S]
        f = sc - torch.logsumexp(ss, dim=1)
        return torch.stack([f, f], dim=1)


class TripletTarget(nn.Module):
    """M3: F(x) = z.zp - z.zn  (zp target prototype, zn local sibling prototype; both unit norm)."""
    def __init__(self, encoder): super().__init__(); self.encoder = encoder
    def forward(self, x, zp, zn):
        z = self.encoder(x)
        f = (z * zp).sum(1) - (z * zn).sum(1)
        return torch.stack([f, f], dim=1)


def _ig(mod):
    return IntegratedGradients(mod, multiply_by_inputs=True)


def attr_dir(C_c, C_sib, u, n_steps=50):
    """M0/M2: signed IG of F=z.u from balanced-sibling baseline to target pseudobulk. C_c,C_sib=[G], u=[128]."""
    mod = CentroidTarget(enc).to(dev).eval(); ig = _ig(mod)
    inp = torch.tensor(C_c[None], dtype=torch.float32, device=dev).requires_grad_(True)
    base = torch.tensor(C_sib[None], dtype=torch.float32, device=dev)
    uu = torch.tensor(u[None], dtype=torch.float32, device=dev)
    a = ig.attribute(inp, baselines=base, target=0, additional_forward_args=(uu,), n_steps=n_steps)
    return a.detach().cpu().numpy().ravel()


def attr_softmax(C_c, C_sib, muc, Msib, tau, n_steps=50):
    """M1: signed IG of the adaptive sibling-softmax margin."""
    mod = SoftmaxTarget(enc, muc, Msib, tau).to(dev).eval(); ig = _ig(mod)
    inp = torch.tensor(C_c[None], dtype=torch.float32, device=dev).requires_grad_(True)
    base = torch.tensor(C_sib[None], dtype=torch.float32, device=dev)
    a = ig.attribute(inp, baselines=base, target=0, n_steps=n_steps)
    return a.detach().cpu().numpy().ravel()


def attr_triplet(Xcells, Xneg, zp, Zneg, n_steps=32, cap=64, rng=None):
    """M3: per target cell IG of F=z.zp - z.zn. Xcells=[Nc,G] target-cell expr, Xneg=[Ns,G] hardest-sibling
    cell expr, zp=[128] target prototype (unit), Zneg=[Ns,128] sibling-cell embeddings (unit). For each sampled
    target cell: zn=nearest sibling cell in latent (its expr = IG baseline). Returns Strength(median +IG),
    Consistency(P(IG>0)), S=Strength*Consistency, and the mean signed IG, each [G]."""
    rng = rng or np.random.default_rng(0)
    Zc = embed(Xcells)                                                          # [Nc,128] unit
    idx = rng.choice(len(Xcells), min(cap, len(Xcells)), replace=False)
    mod = TripletTarget(enc).to(dev).eval(); ig = _ig(mod)
    zp_t = torch.tensor(zp[None], dtype=torch.float32, device=dev)
    acc = []                                                                    # [n, G] signed IG per cell
    for i in idx:
        zi = Zc[i]; nn_j = int(np.argmax(Zneg @ zi))                            # nearest hardest-sibling cell (cosine)
        inp = torch.tensor(Xcells[i][None], dtype=torch.float32, device=dev).requires_grad_(True)
        base = torch.tensor(Xneg[nn_j][None], dtype=torch.float32, device=dev)
        zn_t = torch.tensor(Zneg[nn_j][None], dtype=torch.float32, device=dev)
        a = ig.attribute(inp, baselines=base, target=0, additional_forward_args=(zp_t, zn_t), n_steps=n_steps)
        acc.append(a.detach().cpu().numpy().ravel())
    Aig = np.vstack(acc)                                                        # [n,G]
    strength = np.median(np.clip(Aig, 0, None), axis=0)                         # median positive IG
    consistency = (Aig > 0).mean(0)                                             # P(IG>0)
    return strength * consistency, strength, consistency, Aig.mean(0)


def completeness_dir(C_c, C_sib, u):
    mod = CentroidTarget(enc).to(dev).eval()
    with torch.no_grad():
        x = torch.tensor(np.vstack([C_c, C_sib]), dtype=torch.float32, device=dev)
        f = (mod.encoder(x) * torch.tensor(u, dtype=torch.float32, device=dev)).sum(1).cpu().numpy()
    return f[0] - f[1]


def group_dirs(Z, labels, types, group_of, lineage_of):
    """per-type mean embeddings + per-type sibling baseline dir (M0/M2) + parent dir (M2) + unit centroids (M1).
    Returns dict per type index i: muc[128], Msib_types(list of sib type idx), musib[128], vsub[128], uparent[128]."""
    zc = {i: Z[labels == t].mean(0) for i, t in enumerate(types)}               # per-type mean embedding
    out = {}
    for i, t in enumerate(types):
        g = group_of[t]; sib = [j for j, s in enumerate(types) if s != t and group_of[s] == g]
        if not sib: sib = [j for j in range(len(types)) if j != i]              # singleton fallback: all others
        musib = np.mean([zc[j] for j in sib], 0)
        vsub = zc[i] - musib
        lin = lineage_of[t]; inL = [j for j, s in enumerate(types) if lineage_of[s] == lin]
        outL = [j for j, s in enumerate(types) if lineage_of[s] != lin]
        if outL:
            up = np.mean([zc[j] for j in inL], 0) - np.mean([zc[j] for j in outL], 0)
        else:
            up = vsub.copy()
        uparent = up / (np.linalg.norm(up) + 1e-12)
        out[i] = dict(muc=zc[i], sib=sib, musib=musib, vsub=vsub, uparent=uparent,
                      muc_unit=zc[i] / (np.linalg.norm(zc[i]) + 1e-12),
                      Msib_unit=np.vstack([zc[j] / (np.linalg.norm(zc[j]) + 1e-12) for j in sib]))
    return out


def u_m0(d):
    u = d["vsub"]; return u / (np.linalg.norm(u) + 1e-12)


def u_m2(d):
    v = d["vsub"]; proj = float(v @ d["uparent"]); vperp = v - proj * d["uparent"]
    return vperp / (np.linalg.norm(vperp) + 1e-12), abs(proj), abs(proj) / (np.linalg.norm(v) + 1e-12)
