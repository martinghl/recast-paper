"""The metrics of the ANS scoring benchmark, with the same computation as the ANS repository
(experiments/comparable_score_ranges/comparable_score_range_experiment.py):

  get_lbl_assignment_performance  (hard labels)
    conf_mat = confusion_matrix(y_true, y_pred, normalize='true', labels=label_names)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    f1_val   = f1_score(y_true, y_pred, average='weighted')
    jacc     = jaccard_score(y_true, y_pred, average='weighted')

  get_information_from_scores  (information quantity)
    Pipeline([StandardScaler, LogisticRegression(max_iter=1000, penalty=None)])
    cv = StratifiedKFold(n_splits=10)            # no shuffle, no random_state -> deterministic
    cross_val_score(model, X, y, cv=cv, scoring=met) for met in
        ['balanced_accuracy', 'f1_weighted', 'jaccard_weighted']
    info_quantity = mean of the balanced_accuracy fold scores

  scale_imbalance = info_quantity - balanced_accuracy   (signed)

The functions take the score matrix and labels as arrays instead of reading them from `adata.obs`; the numbers
are the same.  The per-cell argmax (predicted signature) is taken by the caller (`score_df.idxmax(axis=1)`)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score, f1_score,
                             jaccard_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_lbl_assignment_performance(y_true, y_pred, label_names, avg="weighted"):
    """ANS hard-label metrics from per-cell argmax predictions.

    Returns (conf_mat, balanced_accuracy, f1_weighted, jaccard_weighted). `conf_mat` is
    row-normalized (normalize='true') over `label_names` -- as ANS's confusion matrices.
    """
    conf_mat = confusion_matrix(y_true, y_pred, normalize="true", labels=label_names)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_val = f1_score(y_true, y_pred, average=avg)
    jacc = jaccard_score(y_true, y_pred, average=avg)
    return conf_mat, float(bal_acc), float(f1_val), float(jacc)


def get_information_from_scores(X, y, nfold=10, max_iter=1000, n_jobs=1):
    """ANS supervised information quantity: 10-fold logistic-regression CV over the per-cell score
    matrix. `X` = (n_cells, n_signatures) float array; `y` = per-cell true labels.

    Returns {metric: (mean, std)} for balanced_accuracy / f1_weighted / jaccard_weighted. The
    scalar "information quantity" used downstream is agg['balanced_accuracy'][0].

    Implementation note: ANS uses three separate `cross_val_score` calls (one per scoring metric),
    i.e. 3 x nfold estimator fits. This uses a single `cross_validate` with all three scorers so the
    estimator is fit ONCE per fold (nfold fits) and each fold's fitted model is scored by all three
    metrics -- the fold splits (StratifiedKFold, no shuffle -> deterministic) and the fitted models
    are identical to ANS's, so the balanced_accuracy / f1_weighted / jaccard_weighted values are
    bit-identical; only redundant refits are removed. `n_jobs` parallelizes the folds (the
    per-cell logreg with penalty=None on ~80k cells frequently runs to max_iter, so this is the run's
    real cost). Estimator/data are pure sklearn/numpy here -> the joblib workers never import
    signaturescoring, so no AnyRandom shim is needed in them.
    """
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=max_iter, penalty=None)),
    ])
    cv = StratifiedKFold(n_splits=nfold)
    scoring = ["balanced_accuracy", "f1_weighted", "jaccard_weighted"]
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
                         return_train_score=False)
    agg = {met: (float(np.mean(res[f"test_{met}"])), float(np.std(res[f"test_{met}"])))
           for met in scoring}
    return agg


def per_type_recall(y_true, y_pred, labels):
    """Per-class recall (== per-class balanced accuracy).

    ANS neuronal script: recall_score(y_true, y_pred, labels=[group], average='macro') per group.
    Returns {label: recall}.
    """
    out = {}
    for lab in labels:
        out[lab] = float(recall_score(y_true, y_pred, labels=[lab], average="macro"))
    return out


def scale_imbalance(info_quantity, balanced_acc):
    """ANS Score imbalance = info_quantity - balanced_accuracy (signed)."""
    return float(info_quantity - balanced_acc)


def assemble_metrics_row(method, y_true, score_df, label_names, degenerate_info=False):
    """Full metric row for one method.

    score_df: (n_cells x n_signatures) DataFrame, columns == signature names, index == cells.
    y_true:   per-cell true signature label (array/Series aligned to score_df rows).
    label_names: ordered signature names (for the confusion matrix axes).
    degenerate_info: if True (RECAST per-cluster broadcast -> cluster-constant per-cell features),
        the logistic-regression info-quantity is degenerate; it is reported as NaN and scale
        imbalance as NaN, with balanced_accuracy / f1 still computed on the (broadcast) argmax.

    Returns (row_dict, conf_mat, y_pred).
    """
    y_pred = score_df.idxmax(axis=1).values
    conf_mat, bal_acc, f1_val, jacc = get_lbl_assignment_performance(y_true, y_pred, label_names)

    if degenerate_info:
        info_q = np.nan
        info_std = np.nan
        f1_cv = np.nan
        jacc_cv = np.nan
        s_imb = np.nan
    else:
        try:
            agg = get_information_from_scores(score_df.values, y_true)
            info_q, info_std = agg["balanced_accuracy"]
            f1_cv = agg["f1_weighted"][0]
            jacc_cv = agg["jaccard_weighted"][0]
            s_imb = scale_imbalance(info_q, bal_acc)
        except Exception as e:  # noqa: BLE001 -- a CV hiccup must not block the rest (esp. RECAST)
            print(f"  [WARN] info-quantity CV failed for {method}: {type(e).__name__}: {e}",
                  flush=True)
            info_q = info_std = f1_cv = jacc_cv = s_imb = np.nan

    row = {
        "method": method,
        "balanced_acc": bal_acc,
        "f1_weighted": f1_val,
        "jaccard": jacc,
        "info_quantity": info_q,
        "info_quantity_std": info_std,
        "info_f1_weighted": f1_cv,
        "info_jaccard": jacc_cv,
        "scale_imbalance": s_imb,
    }
    return row, conf_mat, y_pred
