"""Sorted.numbers -> categorized_261.csv, the corrected type labels.

`Sorted.numbers` is the categorisation pass over the 340-row labelling set: 261 rows
that have text, sorted by category, plus the 79 `NEEDS_SOURCING` placeholders on a
second tab. Its own note records how the categories were assigned -- a keyword/pattern
classifier, then a manual review of everything left in `Other`.

That provenance matters and is the reason this file is named `categorized`, not
`labelled`. It is **not** the blind annotation README.md asks for: for the ~230 rows
outside `Other` the label came from a classifier, so a head trained on it partly learns
to imitate that classifier, and accuracy measured against it is not ground truth. It is
still a large improvement on `scam_type` -- it disagrees with the mapping on 52% of
rows, and the disagreements are the mapping's known bugs (IRS filed under
`investment and crypto`, the 98-row `other` fallback). Use it as a correction, keep the
blind pass on the plan.

Three defects in the workbook are repaired here rather than in it, so the source file
stays exactly as it was received:

  1. `Romacne`     -- typo, 1 row. Folded into `romance`.
  2. `Real estate` -- a 14th category outside the 13-type taxonomy, 1 row. Its text is
     "Hi, sorry I was in Australia and I'm back..." -- a no-signal opener, not real
     estate at all. Moved to `other`, which the workbook's own note is where
     no-signal openers belong.
  3. The Summary tab is stale: it claims romance 20 / other 30 against an actual
     26 / 23. The counts this script prints are computed from the rows.

    python extract_categorized.py            # writes categorized_261.csv
    python extract_categorized.py --check    # counts only, writes nothing
"""
import argparse
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "Sorted.numbers")
OUT = os.path.join(HERE, "categorized_261.csv")

# id -> corrected category, with the reasoning in the module docstring.
REPAIRS = {"437": "romance", "545": "other"}

# The taxonomy, from dataset/scam_types.yaml. Anything outside it is a defect.
PLAN_TYPES = {
    "government impersonation", "tech support", "bank alert", "delivery and toll",
    "family emergency", "romance", "investment and crypto", "prize and lottery",
    "charity", "medicare and health", "utility shutoff", "job offer", "other",
}


def read_workbook(path=SOURCE):
    """The Categorized tab as (id, category), lowercased and repaired."""
    try:
        from numbers_parser import Document
    except ImportError:
        raise SystemExit("needs numbers-parser: pip install numbers-parser")

    sheet = next(s for s in Document(path).sheets if s.name == "Categorized")
    rows = sheet.tables[0].rows(values_only=True)
    # Two title lines and a blank precede the header, so find the header by content.
    head = next(i for i, r in enumerate(rows) if r and str(r[0]).strip() == "id")
    cols = [str(c).strip() for c in rows[head][:4]]
    df = pd.DataFrame(rows[head + 1:], columns=cols + list(rows[head][4:]))[cols]
    df.columns = ["id", "category", "text", "original_label"]
    df = df[df["id"].notna()].copy()

    # numbers-parser hands back floats for numeric cells; the corpus keys on strings.
    df["id"] = df["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["category"] = df["category"].astype(str).str.strip().str.lower()

    repaired = df["id"].isin(REPAIRS)
    df.loc[repaired, "category"] = df.loc[repaired, "id"].map(REPAIRS)
    return df[["id", "category"]], int(repaired.sum())


def main(check=False):
    df, n_repaired = read_workbook()
    stray = sorted(set(df["category"]) - PLAN_TYPES)
    if stray:
        raise SystemExit(f"categories outside the taxonomy after repair: {stray}")

    counts = df["category"].value_counts()
    print(f"{len(df)} rows, {len(counts)} categories, {n_repaired} repaired")
    print(counts.to_string())

    missing = sorted(PLAN_TYPES - set(df["category"]) - {"other"})
    if missing:
        print(f"\nstill no rows at all: {', '.join(missing)}")

    if check:
        return
    df.sort_values("id", key=lambda s: s.astype(int)).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="report counts, write nothing")
    main(**vars(p.parse_args()))
