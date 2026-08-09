"""Fill the columns merge_datasets.py writes blank, plus the stage-2 pair.

The merge cannot produce any of them -- `split` needs the group-aware splitter and the
rest need trained models -- so they are filled here, in place.

  python annotate.py                # qwen_feat_clean, the recommended arm
  python annotate.py --arm qwen_feat
  python annotate.py --arm minilm_feat --types    # also fill the stage-2 columns

Runs last: merge -> features -> embed -> run_arms -> annotate. Re-running the merge
blanks the columns; re-run this to refill.

  split            train / val / test
  model_pred       stage 1, hard 0/1 at the arm's validation-fitted threshold
  scam_type_true   stage 2 label
  scam_type_pred   stage 2 prediction from the distilled head

Two warnings that apply to every one of the model columns. They are filled for **every**
row, including the ones the model trained on, where they are in-sample and meaningless
-- score on `split == "test"` only. And `scam_type_true` is a *label* while
`scam_type_pred` is a *guess*: they sit two columns apart and the whole reason
labeling/ exists is that this project already once mistook a derived label for
annotation. Never train on the `_pred` column.
"""
import argparse, os, pickle

import pandas as pd

import modeling as M
import run_arms as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = M.DATASET_CSV          # written back in place, in dataset/
RESULTS = M.RESULTS

# The correction pass over the labelling set. Applied on top of the derived scam_type,
# which labeling/README.md documents as buggy on 136 of the 261 rows it covers.
CORRECTED = os.path.join(ROOT, "labeling", "categorized_261.csv")

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


def scam_type_columns(work, X, arm):
    """(true, pred) stage-2 columns, or (None, None) when no head is fitted.

    `true` is the derived scam_type with labeling/categorized_261.csv applied over it,
    blank where nothing knows. `pred` is the distilled head, filled only where stage 1
    flags the row -- the head is trained on flagged messages and its answer on an
    unflagged one has no meaning, so leaving it blank is the honest value.
    """
    head_p = os.path.join(RESULTS, f"type_head_{arm}.pkl")
    if not os.path.exists(head_p):
        return None, None
    with open(head_p, "rb") as fh:
        head = pickle.load(fh)

    true = work["scam_type"].astype(str).copy()
    n_fixed = 0
    if os.path.exists(CORRECTED):
        fix = pd.read_csv(CORRECTED)
        m = dict(zip(fix["id"].astype(str), fix["category"].astype(str)))
        mapped = work["id"].astype(str).map(m)
        n_fixed = int((mapped.notna() & (mapped != true)).sum())
        true = mapped.fillna(true)
    print(f"scam_type_true: {int((true != '').sum())} labelled rows "
          f"({n_fixed} corrected from categorized_261.csv)")

    pred = pd.Series([""] * len(work), index=work.index, dtype=object)
    flagged = work["_flagged"].to_numpy()
    if flagged.any():
        pred.loc[flagged] = head["model"].predict(X[flagged])
    print(f"scam_type_pred: {int(flagged.sum())} flagged rows predicted, "
          f"{len(work) - int(flagged.sum())} left blank (stage 1 did not flag them)")
    return true, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=DEFAULT_ARM, choices=REPLAYABLE)
    ap.add_argument("--types", action="store_true",
                    help="also fill scam_type_true and scam_type_pred")
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

    if a.types:
        work["_flagged"] = pred.astype(bool)
        true, tpred = scam_type_columns(work, X, a.arm)
        if true is None:
            raise SystemExit(f"no type_head_{a.arm}.pkl -- run: "
                             f"python ../app-backend/distill.py --train --arm {a.arm}")
        raw["scam_type_true"] = true.values
        raw["scam_type_pred"] = tpred.values

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
