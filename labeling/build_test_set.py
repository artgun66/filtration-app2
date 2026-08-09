"""A blind labelling set from the test split's untyped rows.

`dataset_340.csv` samples SmishTank rows, which carry a category and so already have a
mapped scam_type. This builds the set that closes a different gap: the **test split**
has 10,120 rows and only 199 carry a type, so there is no way to score both stages
end-to-end on held-out data. Every stage-2 number in this project is measured on the
type head's own split, drawn from a pool that is 26% test rows already.

Labelling everything is not the answer -- 9,921 test rows have no type. What matters is
the population stage 2 actually meets, which is everything stage 1 flags, plus the scams
it misses so recall stays measurable:

    true scam, flagged   144      stage 2 sees these
    ham,       flagged    33      stage 2 sees these too, and should say `not a scam`
    true scam, missed     11      stage 1's misses, needed for an honest denominator
                         ----
                         188

Written blind, same as the other sets here: text only, no mapped label, no model
prediction, no probability. The key is a separate file. Reading it first turns
annotation into review, and people overturn far less often than they disagree.

    python build_test_set.py
"""
import csv
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scam-classification"))

import modeling as M                                    # noqa: E402
import run_arms as R                                    # noqa: E402
from annotate import threshold_for                      # noqa: E402

ARM = "minilm_feat"          # the shipped arm: its mistakes are the ones that matter
BLIND = os.path.join(HERE, "test_unlabelled.csv")
KEY = os.path.join(HERE, "test_unlabelled_key.csv")

BLIND_COLS = ["id", "text", "label", "second_label", "confidence", "flag", "note"]
KEY_COLS = ["id", "split", "is_scam", "stage1_prob", "stage1_flagged", "case"]


def main():
    df = M.load()
    with open(os.path.join(M.RESULTS, f"model_{ARM}.pkl"), "rb") as fh:
        model = pickle.load(fh)["model"]

    X, _ = R.feature_matrix(ARM, df, df["id"].to_numpy())
    prob = model.predict_proba(X)[:, 1]
    thr = threshold_for(ARM)

    df = df.assign(_prob=prob, _flagged=prob >= thr)
    pool = df[(df["split"] == "test")
              & (df["scam_type"].astype(str) == "")
              & ((df["scam"] == 1) | df["_flagged"])].copy()

    def case(r):
        if r["scam"] == 1:
            return "true scam, flagged" if r["_flagged"] else "true scam, missed"
        return "ham, flagged"

    pool["case"] = pool.apply(case, axis=1)
    # Shuffled: labelling all 144 true scams and then 33 false positives in a block
    # turns into pattern-matching on position instead of on content.
    pool = pool.sample(frac=1.0, random_state=M.SEED)

    with open(BLIND, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, BLIND_COLS)
        w.writeheader()
        for _, r in pool.iterrows():
            w.writerow({"id": r["id"], "text": r["text"], "label": "",
                        "second_label": "", "confidence": "", "flag": "", "note": ""})

    with open(KEY, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, KEY_COLS)
        w.writeheader()
        for _, r in pool.iterrows():
            w.writerow({"id": r["id"], "split": r["split"], "is_scam": int(r["scam"]),
                        "stage1_prob": round(float(r["_prob"]), 6),
                        "stage1_flagged": int(r["_flagged"]), "case": r["case"]})

    print(f"arm {ARM} at threshold {thr:.6f}\n")
    print(pool["case"].value_counts().to_string())
    print(f"\n{len(pool)} rows")
    print(f"  {os.path.relpath(BLIND, ROOT)}   blind, for the labeller")
    print(f"  {os.path.relpath(KEY, ROOT)}   do not open until done")


if __name__ == "__main__":
    main()
