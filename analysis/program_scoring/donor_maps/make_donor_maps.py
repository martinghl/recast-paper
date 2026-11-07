#!/usr/bin/env python
"""Source and check of the donor maps used by donor_bootstrap.py.

donor_bootstrap.py joins `<parent>_donor.tsv` onto the benchmark cells and resamples those labels (intervals of
Fig. 6a).  This script names the object and column each map comes from and exports them again to check the stored
files.  The objects are on the CPU server, so --verify runs the export there and copies the TSVs back.

  python make_donor_maps.py --verify            # all parents
  python make_donor_maps.py --verify --only zhao lawlor

`donor` is the resampling cluster: an individual for every parent except zhao, whose object has no subject column,
only `sample` = <individual> x <tissue> (3 individuals x blood / liver / spleen = 9 units); see ZHAO_UNIT below."""
import argparse, os, re, subprocess, sys, tempfile

HOST = "cpu_server"
RD   = "/path/to/cpu_server/analysis/marker_methods/data/real_data"
DM   = os.path.dirname(os.path.abspath(__file__))
RSCRIPT = "/path/to/envs/marker_methods/bin/Rscript"
PYREM   = "/path/to/envs/marker_methods/bin/python"

# parent -> (kind, remote path, column, what one label means)
SOURCES = {
    "dominguez_conde": ("rds",  f"{RD}/dominguez_conde.rds", "donor_id",       "individual"),
    "ren":             ("rds",  f"{RD}/ren.rds",             "donor_id",       "individual"),
    "stephenson":      ("rds",  f"{RD}/stephenson.rds",      "donor_id",       "individual"),
    "yoshida":         ("rds",  f"{RD}/yoshida.rds",         "donor_id",       "individual"),
    "zhao":            ("rds",  f"{RD}/zhao.rds",            "sample",
                        "individual (COLLAPSED from 9 donor x tissue samples -- see COLLAPSE)"),
    "lawlor":          ("rds",  f"{RD}/lawlor.rds",          "islet unos id",  "individual (organ donor)"),
    # pbmcbmn's SCE was built by marker_methods/code/prepare_hao_pbmc.R from this h5ad and kept only label +
    # sizeFactor, so the donor column has to come from the h5ad the prep script read.
    "pbmcbmn":         ("h5ad", "/path/to/cpu_server/analysis/pp_pbmc_b_mono_nk_scored.h5ad",
                        "donor", "individual (8 multiplexed across 13 lanes)"),
}

ZHAO_UNIT = """zhao: the source object has no subject column -- `sample` is the only grouping and its
9 levels are 'donor {1,2,3} {blood,liver,spleen}', i.e. 3 individuals sampled in 3 tissues.  Treating
those 9 as donors would count within-individual tissue variation as between-individual variation and
make the interval anti-conservative.  zhao is therefore collapsed to its 3 individuals, which puts
it below the 5-donor floor: both zhao tasks carry a point estimate and no interval, on the same rule
that already applies to Baron (4 donors) and Pelka (2).  The stored map holds the collapsed label,
so COLLAPSE below is applied to the upstream column before --verify compares."""

# parent -> how the stored label is derived from the upstream column (identity where absent)
COLLAPSE = {"zhao": lambda v: re.sub(r"\s+(blood|liver|spleen)$", "", v, flags=re.I)}

R_TMPL = r"""
suppressMessages(library(SingleCellExperiment))
s <- readRDS('%(path)s'); cd <- colData(s)
stopifnot('%(col)s' %%in%% colnames(cd))
df <- data.frame(cell=colnames(s), donor=as.character(cd[['%(col)s']]), check.names=FALSE)
write.table(df, '%(out)s', sep='\t', quote=FALSE, row.names=FALSE)
"""

PY_TMPL = r"""
import h5py, numpy as np
f = h5py.File('%(path)s', 'r')
def col(k):
    g = f['obs'][k]
    if isinstance(g, h5py.Group):
        return np.array(g['categories']).astype(str)[np.array(g['codes'])]
    return np.array(g).astype(str)
assert '%(col)s' in f['obs'], 'no %(col)s in obs'
with open('%(out)s', 'w') as o:
    o.write('cell\tdonor\n')
    for c, v in zip(col('_index'), col('%(col)s')):
        o.write('%%s\t%%s\n' %% (c, v))
"""


def _ssh(script, runner, remote_script):
    """Write the script remotely and run it.  Output goes to a FILE, never stdout: the host prints a
    login banner that would be concatenated onto the first data line."""
    subprocess.run(["ssh", HOST, f"cat > {remote_script}"], input=script.encode(), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ssh", HOST, f"{runner} {remote_script}"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def verify(parents):
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for p in parents:
            kind, path, col, unit = SOURCES[p]
            stored = os.path.join(DM, f"{p}_donor.tsv")
            if not os.path.exists(stored):
                print(f"  {p:<16} NO STORED MAP"); bad += 1; continue
            out = f"/path/to/cpu_server/.verify_{p}_donor.tsv"
            args = dict(path=path, col=col, out=out)
            if kind == "rds":
                _ssh(R_TMPL % args, RSCRIPT, f"/path/to/cpu_server/.verify_{p}.R")
            else:
                _ssh(PY_TMPL % args, PYREM, f"/path/to/cpu_server/.verify_{p}.py")
            local = os.path.join(tmp, f"{p}.tsv")
            subprocess.run(["scp", "-q", f"{HOST}:{out}", local], check=True)
            import pandas as pd
            src = pd.read_csv(local, sep="\t", dtype=str)
            if p in COLLAPSE:
                src["donor"] = src.donor.map(COLLAPSE[p])
            shp = pd.read_csv(stored, sep="\t", dtype=str)
            m = src.merge(shp, on="cell", how="outer", suffixes=("_src", "_stored"), indicator=True)
            miss = int((m._merge != "both").sum())
            diff = int((m.donor_src != m.donor_stored).sum())
            ok = miss == 0 and diff == 0 and len(src) == len(shp)
            bad += not ok
            print(f"  {p:<16} {len(shp):>6} cells  {shp.donor.nunique():>3} labels  "
                  f"{col:<14} {'EXACT' if ok else f'MISMATCH ({miss} unmatched, {diff} differing)'}"
                  f"   unit = {unit}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    ps = a.only or list(SOURCES)
    unknown = [p for p in ps if p not in SOURCES]
    if unknown:
        sys.exit(f"unknown parent(s): {unknown}")
    if not a.verify:
        print(__doc__)
        for p, (k, path, col, unit) in SOURCES.items():
            print(f"  {p:<16} {k:<5} {col:<14} {unit:<38} {path}")
        print("\n" + ZHAO_UNIT)
        sys.exit(0)
    n = verify(ps)
    print(f"\n{len(ps) - n}/{len(ps)} donor maps reproduce their upstream column exactly.")
    sys.exit(1 if n else 0)
