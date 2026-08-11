"""Readable per-split slices of the corpus.

combined_sms_dataset.csv is 88,542 rows by 104 columns, and the split lives in column
26 as a bare word -- fine for code, unreadable in an editor. This writes one file per
split with only the columns a person would look at.

Views, not data: regenerable in seconds, gitignored, and never read by anything else.
Edit the corpus, not these.

    python split_views.py                # val and test
    python split_views.py --all          # including train (68,302 rows)
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "combined_sms_dataset.csv")
OUT = os.path.join(HERE, "views")

# Enough to judge a row by eye. `scam` is the stage-1 target, `model_pred` its
# prediction; `scam_type_true` is a label and `scam_type_pred` is a guess.
COLS = ["id", "split", "label", "scam", "model_pred",
        "scam_type", "scam_type_true", "scam_type_pred", "source", "text"]


def main(all_splits=False):
    df = pd.read_csv(SRC, keep_default_na=False, dtype=str)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns {missing} -- run: "
                         f"python ../scam-classification/annotate.py --types")

    os.makedirs(OUT, exist_ok=True)
    wanted = ["train", "val", "test"] if all_splits else ["val", "test"]
    for s in wanted:
        part = df[df["split"] == s][COLS]
        path = os.path.join(OUT, f"{s}.csv")
        part.to_csv(path, index=False)
        typed = int((part["scam_type_true"] != "").sum())
        print(f"{s:<6} {len(part):>6} rows, {int((part['scam'] == '1').sum()):>5} scam, "
              f"{typed:>4} with a scam type  ->  views/{s}.csv")

    if not all_splits:
        print("\ntrain omitted (68,302 rows) -- pass --all if you want it")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all", dest="all_splits", action="store_true",
                   help="also write train.csv")
    main(**vars(p.parse_args()))
