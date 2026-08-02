"""Splitting, metrics and hyper-parameter search for the suspicious-SMS classifier.

Target: y = 1 for suspicious (spam or smishing), 0 for ham -- the dataset's `scam`
column. Ported from analysis_final.ipynb with three defects from that notebook
fixed; each is called out where it occurs, since copying it verbatim would silently
optimise the wrong number.

`python modeling.py` runs the metric self-test.
"""
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASET_CSV = os.path.join(ROOT, "dataset", "combined_sms_dataset.csv")
RESULTS = os.path.join(HERE, "results")

with open(os.path.join(HERE, "training.yaml"), encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)

SEED = CFG["seed"]
BOUNDS = {k: tuple(v) for k, v in CFG["lightgbm"]["bounds"].items()}
INT_PARAMS = tuple(CFG["lightgbm"]["int_params"])
MIN_RECALL = CFG["threshold"]["min_recall"]


# ---------------------------------------------------------------- data
def load(src=DATASET_CSV):
    """The dataset with the training target attached.

    y is the `scam` column: 1 for spam OR smishing, 0 for ham. Taken by name rather
    than recomputed, so the target the models train on is the one the dataset
    publishes. Asserted against the label, since a silent mismatch here would move
    every number downstream.
    """
    df = pd.read_csv(src, keep_default_na=False, dtype={"text": str})
    df["y"] = df["scam"].astype(int)
    assert (df["y"] == (df["label"] != "ham")).all(), \
        "the scam column disagrees with the label column"
    return df


def make_splits(df, seed=SEED):
    """Group-aware 3-way split with clean-only validation and test.

    Two rules, both load-bearing:
      * no dup_group straddles a split -- smishing is templated, and a row-level
        split would put near-identical messages on both sides;
      * val and test use only groups where every row is clean-labelled, so scores
        measure detection, not agreement with the keyword rule that labelled
        Smishing_Dataset. Those rows still train.
    """
    group, clean = CFG["split"]["group_column"], CFG["split"]["clean_column"]
    # min() over the group: 0 if any row in the group came from the keyword source
    eligible = df.groupby(group)[clean].transform("min") == 1
    elig = df[eligible]

    sgkf = StratifiedGroupKFold(n_splits=CFG["split"]["n_folds"],
                                shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(elig, elig["y"], groups=elig[group])]
    test_ids = set(elig.iloc[folds[0]]["id"])
    val_ids = set(elig.iloc[folds[1]]["id"])

    where = np.where(df["id"].isin(test_ids), "test",
                     np.where(df["id"].isin(val_ids), "val", "train"))
    return pd.Series(where, index=df.index, name="split")


def describe_splits(df, split):
    """Print split composition and assert the two split invariants.

    The clean pool is overwhelmingly NUS ham, so test lands near 350 positives at a
    ~3.5% positive rate against ~42% in train. Two consequences: recall's bootstrap
    CI is about +/- 2 points, so report the interval; and precision/accuracy move
    with prevalence, so they are not comparable across splits. Recall, specificity
    and AUC are.
    """
    tab = pd.crosstab(split, df["label"])
    tab["rows"] = tab.sum(axis=1)
    tab["positives"] = df.groupby(split)["y"].sum()
    tab["positive_rate"] = df.groupby(split)["y"].mean().round(4)
    print(tab.to_string())
    n_pos = int(df.loc[split == "test", "y"].sum())
    print(f"\ntest positives: {n_pos} -- recall CI on this set is about +/- "
          f"{1.96 * (0.95 * 0.05 / n_pos) ** 0.5 * 100:.1f} points at 95% recall")
    # the two split invariants, asserted rather than eyeballed
    assert not ((split != "train") & (df["is_clean_label"] == 0)).any(), \
        "keyword-labelled rows reached val/test"
    spans = df.assign(_s=split.values).groupby("dup_group")["_s"].nunique()
    assert spans.max() == 1, f"{int((spans > 1).sum())} dup_groups straddle a split"
    return tab


# ---------------------------------------------------------------- metrics
def binary_metrics(y_true, y_prob, threshold):
    """Confusion-matrix metrics at a threshold.

    recall = TP/(TP+FN), specificity = TN/(TN+FP). Spelled out because
    analysis_final.ipynb cell 8 has both transposed -- it computes precision and NPV
    under those names, and the plan's target is recall. The self-test below proves
    the two forms differ.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "ap": average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def pick_threshold(y_true, y_prob, min_recall=MIN_RECALL):
    """Highest threshold whose recall still clears the floor.

    Highest, not lowest: every threshold below some t* satisfies the constraint, and
    the lowest is 0 -- recall 1.0 at specificity 0. The highest maximises
    specificity. Chosen on validation only; test is scored at whatever comes out.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    pos = np.sort(y_prob[y_true == 1])
    if len(pos) == 0:
        return 0.5
    # the k-th smallest positive score is the largest threshold keeping >= min_recall
    k = int(np.floor((1.0 - min_recall) * len(pos)))
    k = min(max(k, 0), len(pos) - 1)
    return float(pos[k])


def bootstrap_ci(y_true, y_prob, threshold, metric="recall",
                 n=CFG["bootstrap"]["n"], seed=SEED):
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx])) < 2:
            continue
        vals.append(binary_metrics(y_true[idx], y_prob[idx], threshold)[metric])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


# ---------------------------------------------------------------- LightGBM + Bayesian opt
# BOUNDS and INT_PARAMS come from training.yaml, at the top of this file.
def _params(seed=SEED, **kw):
    p = {k: (int(v) if k in INT_PARAMS else v) for k, v in kw.items()}
    p.update(random_state=seed, verbose=-1, n_jobs=CFG["lightgbm"]["n_jobs"])
    return p


def fit_lgbm(X_train, y_train, params, seed=SEED):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**_params(seed=seed, **params)).fit(X_train, y_train)


def tune_lgbm(X_train, y_train, X_val, y_val,
              init_points=CFG["bayes_opt"]["init_points"],
              n_iter=CFG["bayes_opt"]["n_iter"], seed=SEED):
    """Bayesian search over BOUNDS, scored by validation AUC.

    No SHAP inside the objective (the notebook's version explains the whole dataset
    every iteration and never finishes) and no test score printed during the search,
    so the run cannot be steered by it.
    """
    from bayes_opt import BayesianOptimization

    def objective(**kw):
        clf = fit_lgbm(X_train, y_train, kw, seed=seed)
        return roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

    bo = BayesianOptimization(objective, BOUNDS, random_state=seed, verbose=0)
    bo.maximize(init_points=init_points, n_iter=n_iter)
    return bo.max["params"], bo.max["target"]


# ---------------------------------------------------------------- reporting
def evaluate(name, clf, splits, min_recall=MIN_RECALL, ci=False):
    """Tune the threshold on val, then report train/val/test at that threshold."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    p_va = clf.predict_proba(Xva)[:, 1]
    thr = pick_threshold(yva, p_va, min_recall)

    rows = {}
    for split, X, y in (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)):
        p = p_va if split == "val" else clf.predict_proba(X)[:, 1]
        rows[split] = binary_metrics(y, p, thr)
        if ci and split == "test":
            lo, hi = bootstrap_ci(y, p, thr, "recall")
            rows[split]["recall_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    out = pd.DataFrame(rows).T
    out.insert(0, "arm", name)
    return out


if __name__ == "__main__":
    # Verification step 4 from the plan: check the metric formulas by hand on a case
    # small enough to count on your fingers, since this is exactly what the template
    # got wrong.
    # FP and FN must differ, or the correct and transposed formulas coincide and the
    # test proves nothing. Here FP=2, FN=1.
    y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    p = [.9, .8, .7, .6, .2, .8, .7, .3, .1, .1]     # thr .5 -> TP=4 FN=1 FP=2 TN=3
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (4, 1, 2, 3), m
    assert abs(m["recall"] - 4 / 5) < 1e-9, m["recall"]            # TP/(TP+FN)
    assert abs(m["specificity"] - 3 / 5) < 1e-9, m["specificity"]  # TN/(TN+FP)
    assert abs(m["precision"] - 4 / 6) < 1e-9, m["precision"]

    # the template's formulas, to show they differ and are not interchangeable
    wrong_recall = m["tp"] / (m["tp"] + m["fp"])          # = precision, 0.667 not 0.800
    wrong_spec = m["tn"] / (m["tn"] + m["fn"])            # = NPV,       0.750 not 0.600
    assert abs(wrong_recall - m["recall"]) > 0.1
    assert abs(wrong_spec - m["specificity"]) > 0.1

    # threshold picker: with 5 positives a 0.95 floor cannot drop any of them, so the
    # answer is the lowest positive score; a 0.75 floor may drop one.
    assert abs(pick_threshold(y, p, 0.95) - 0.2) < 1e-9
    assert abs(pick_threshold(y, p, 0.75) - 0.6) < 1e-9
    assert binary_metrics(y, p, pick_threshold(y, p, 0.75))["recall"] >= 0.75

    print("metrics self-test passed")
    print(f"  correct recall      {m['recall']:.4f}   template's formula {wrong_recall:.4f}")
    print(f"  correct specificity {m['specificity']:.4f}   template's formula {wrong_spec:.4f}")
