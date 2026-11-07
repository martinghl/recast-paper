"""Match a RECAST panel's subtype name to a DE method's own cluster label.

Several methods of the marker_methods harness build their result tables with R semantics that pass subtype names
through `make.names()`, so "Naive B cells" comes back as "Naive.B.cells".  pick() looks a subtype up by its name and,
if that fails, by the normalized name, in which every run of non-alphanumeric characters becomes a dot (what
make.names does to spaces, slashes and hyphens).  Checked to be collision-free on every RECAST panel and every
method's cluster set of the benchmark.  A method that returned no rows for a subtype (e.g. seurat_roc on tf_b_yosh
"Memory B cells") still gets [] and a recall of 0."""
import re

_norm = lambda s: re.sub(r"[^0-9A-Za-z]+", ".", str(s)).strip(".")


def index(genesets):
    """{cluster -> genes} augmented with a normalised-name alias for each cluster."""
    out = dict(genesets)
    for c, g in genesets.items():
        out.setdefault(_norm(c), g)
    return out


def pick(genesets, cluster):
    """Ranked gene list for `cluster`, tolerating make.names-style renaming. [] if truly absent."""
    if cluster in genesets:
        return genesets[cluster]
    return genesets.get(_norm(cluster), [])
