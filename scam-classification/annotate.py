"""Fill the two columns merge_datasets.py writes blank: `split` and `model_pred`.

The merge cannot produce either -- `split` needs the group-aware splitter, and
`model_pred` needs a trained model -- so they are filled here, in place.

  python annotate.py                # qwen_feat_clean, the recommended arm
  python annotate.py --arm qwen_feat

Runs last: merge -> features -> embed -> run_arms -> annotate. Re-running the merge
blanks both columns; re-run this to refill.

`model_pred` is a hard 0/1 at the arm's validation-fitted threshold, not a
probability, and it is filled for every row including the ones the model trained on,
where it is in-sample. Score it on `split == "test"` only.
"""
import argparse, os, pickle

import pandas as pd

import modeling as M
import run_arms as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = M.DATASET_CSV          # written back in place, in dataset/
RESULTS = M.RESULTS

# Arms whose fitted model can be replayed on new rows -- see training.yaml. keyword
# has no model; ann is not pickled; tfidf's pickle holds the classifier but not the
# vectorizer, so its feature space cannot be rebuilt from disk.
REPLAYABLE = M.CFG["replayable"]
DEFAULT_ARM = M.CFG["default_arm"]


def threshold_for(arm):
    """The operating point run_arms.py chose on validation, as recorded."""
    m = pd.read_csv(os.path.join(RESULTS, "arm_metrics.csv"))
    row = m[(m["arm"] == arm) & (m["split"] == "test")]
    if row.empty:
        raise SystemExit(f"no arm_metrics.csv row for {arm} -- run: python run_arms.py --arms {arm}")
    return float(row["threshold"].iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=DEFAULT_ARM, choices=REPLAYABLE)
    a = ap.parse_args()

    # dtype=str keeps every other column exactly as the merge wrote it -- no ints
    # turning into floats, no blanks turning into NaN, on the way back out.
    raw = pd.read_csv(OUT, dtype=str, keep_default_na=False)
    work = raw.copy()
    for c in ("id", "dup_group", "is_clean_label"):
        work[c] = raw[c].astype(int)
    work["y"] = (raw["label"] != "ham").astype(int)

    split = M.make_splits(work)
    M.describe_splits(work, split)

    model_p = os.path.join(RESULTS, f"model_{a.arm}.pkl")
    if not os.path.exists(model_p):
        raise SystemExit(f"no {model_p} -- run: python run_arms.py --arms {a.arm}")
    with open(model_p, "rb") as fh:
        clf = pickle.load(fh)["model"]

    ids = work["id"].to_numpy()
    X, _ = R.feature_matrix(a.arm, work, ids)
    thr = threshold_for(a.arm)
    prob = clf.predict_proba(X)[:, 1]
    pred = (prob >= thr).astype(int)

    raw["split"] = split.values
    raw["model_pred"] = pred
    raw.to_csv(OUT, index=False)

    print(f"\nmodel_pred: arm {a.arm} at its validation threshold {thr:.6f}")
    tab = pd.crosstab(split, pred).rename_axis(columns="model_pred")
    print(tab.to_string())
    te = split == "test"
    m = M.binary_metrics(work.loc[te, "y"], prob[te.to_numpy()], thr)
    print(f"\ntest recall {m['recall']:.3f}  specificity {m['specificity']:.3f}  "
          f"precision {m['precision']:.3f}   (should match results/arm_metrics.csv)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
