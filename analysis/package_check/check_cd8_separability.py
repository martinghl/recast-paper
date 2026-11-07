"""Fig. 2a: k-nearest-neighbour separability of the four Zhao CD8 subtypes in the SCimilarity embedding.

The embedding is the package encoder's (SCimilarityEncoder.embed on raw counts); the classifier is the original
one (subtype_benchmark/encoders_recast.py: L2-normalized embeddings, 15 neighbours, cosine metric,
5-fold cross_val_predict). Compared with subtype_benchmark/results/fm_finestate_knn_ALL_multi_v3.csv.

    python check_cd8_separability.py
Writes cd8_separability.csv."""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

import common as C

STATES = ["CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8"]


def main():
    enc = C.scimilarity()
    _, obs = C.attribution_obs(C.ATLASES, "zhao")
    a, kept = C.lineage_adata(C.ATLASES, "zhao", obs["label"].astype(str).to_numpy(), STATES, min_cells=60)
    y = a.obs["state"].to_numpy()
    Z = normalize(np.asarray(enc.embed(a.X), dtype="float32"))
    yhat = cross_val_predict(KNeighborsClassifier(15, metric="cosine"), Z, y, cv=5)
    acc, f1 = round(float(accuracy_score(y, yhat)), 3), round(float(f1_score(y, yhat, average="macro")), 3)
    stored = pd.read_csv(os.path.join(C.FSB, "results", "fm_finestate_knn_ALL_multi_v3.csv"))
    stored = stored[(stored.FM == "SCimilarity") & (stored.lineage == "zhao_CD8")].iloc[0]
    print(f"{a.n_obs} cells, {len(kept)} subtypes: accuracy {acc} (original {stored.state_knn_acc}), "
          f"macro-F1 {f1} (original {stored.state_macroF1})")
    C.assert_package_only()
    C.write(pd.DataFrame([dict(n_cells=a.n_obs, acc=acc, acc_stored=stored.state_knn_acc, macro_f1=f1,
                               macro_f1_stored=stored.state_macroF1)]), "cd8_separability.csv")


if __name__ == "__main__":
    main()
