"""Add the missing `id` column to eval CSVs written before scam_type_prompt.py kept it.

The evaluation runs cost hours of GPU time, so the fix is to repair the files rather
than re-run them. Ids are recovered by joining on message text -- the same fragile join
the fix removes, done once here under supervision instead of by every consumer forever.

Refuses rather than guesses: a text that matches no corpus row, or more than one, is
reported and left with an empty id.

    python backfill_ids.py            # report only
    python backfill_ids.py --write
"""
import argparse
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASET = os.path.join(ROOT, "dataset", "combined_sms_dataset.csv")
RESULTS = os.path.join(HERE, "results")


def main(write=False):
    corpus = pd.read_csv(DATASET, keep_default_na=False, dtype={"text": str},
                         usecols=["id", "text"])
    counts = corpus["text"].value_counts()
    unique = corpus[corpus["text"].map(counts) == 1]
    lookup = dict(zip(unique["text"], unique["id"]))
    print(f"corpus {len(corpus)} rows, {len(lookup)} with a unique text\n")

    for path in sorted(glob.glob(os.path.join(RESULTS, "scam_type_eval_*.csv"))):
        df = pd.read_csv(path, keep_default_na=False, dtype={"text": str})
        name = os.path.basename(path)
        if "id" in df.columns:
            print(f"{name:<34} already has ids, skipped")
            continue

        ids = df["text"].map(lookup)
        hit = int(ids.notna().sum())
        print(f"{name:<34} {hit}/{len(df)} joined"
              + ("" if hit == len(df) else f"   {len(df) - hit} UNMATCHED"))
        if not write:
            continue

        df.insert(0, "id", ids.fillna("").astype(str).str.replace(r"\.0$", "",
                                                                  regex=True))
        df.to_csv(path, index=False)
        print(f"{'':<34} wrote {name}")

    if not write:
        print("\nreport only -- pass --write to modify the files")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true", help="modify the files in place")
    main(**vars(p.parse_args()))
