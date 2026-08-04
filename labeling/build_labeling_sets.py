"""Build the blind annotation sets for stage 2.

The stage-2 "ground truth" in combined_sms_dataset.csv is not annotation -- it is
SmishTank's own category run through the mapping in dataset/scam_types.yaml. That
mapping has two known failure modes:

  1. category_map wins before the brand is consulted, so 20 IRS-branded messages
     filed by SmishTank under Finance/Crypto become `investment and crypto`.
  2. scam_types.py falls through to `other` when a brand_decides row carries a brand
     that is not in the yaml lists -- 98 rows, 85 of them with a blank brand cell.
     Those are then excluded from the headline accuracy, so they are both mislabelled
     and unmeasured.

This script samples the rows worth hand-labelling and writes them blind: text only,
no mapped label and no model prediction. Those live in answer_key.csv and are joined
back by id after annotation. Seeded with 200, the same seed the rest of the project
uses, so re-running reproduces the identical sets.

    python labeling/build_labeling_sets.py
"""

import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset.scam_types import BRAND_DECIDES, SCAM_TYPES, OTHER  # noqa: E402

DATASET = os.path.join(ROOT, "dataset", "combined_sms_dataset.csv")
EVAL = os.path.join(ROOT, "scam-type-classification", "results",
                    "scam_type_eval_choice.csv")
SEED = 200

# How many of each type to label, and why. The counts are not uniform: 10 per class
# is enough to tell a broken class (~0.50) from a working one (~0.85) but gives a
# +/-25pp interval, while 69 per class for +/-10pp would cost ~900 labels. 25 is the
# knee of the curve, so classes in dispute get 25-30 and classes the model already
# handles get 13 to confirm rather than re-measure.
PLAN_340 = [
    # (bucket, n, note)
    ("__fallback__",             98, "mislabelled `other` AND unscored today"),
    ("investment and crypto",    30, "mapping known-broken: IRS -> crypto"),
    ("tech support",             30, "model known-broken: 0.431, Netflix/Amazon"),
    ("bank alert",               25, "shares the boundary tech support fails on"),
    ("government impersonation", 25, "absorbs the crypto mislabels"),
    ("delivery and toll",        13, "1.000 at n=180 -- confirm, do not re-measure"),
    ("romance",                  13, "already 0.877"),
    ("prize and lottery",        13, "already 0.772"),
    ("job offer",                13, "already 0.760"),
    ("family emergency",         20, "zero coverage -- must be sourced externally"),
    ("charity",                  20, "zero coverage -- must be sourced externally"),
    ("Medicare and health",      20, "zero coverage -- must be sourced externally"),
    ("utility shutoff",          20, "1 row in the corpus -- 19 must be sourced"),
]

# The minimum-viable subset: every row in it is also in the 340 set (same ids), so
# labels done here carry straight over if you go on to finish the full set.
PLAN_240 = {
    "__fallback__":             98,
    "investment and crypto":    30,
    "tech support":             30,
    "family emergency":         20,
    "charity":                  20,
    "Medicare and health":      20,
    "utility shutoff":          20,
    "government impersonation":  2,   # pads 238 -> 240
}

# What the annotator fills in. `label` and `second_label` take a type name from
# type_guide.md; confidence is high/med/low; flag marks rows that are not scoreable
# at all (truncated screenshot, not actually a scam).
BLIND_COLS = ["id", "text", "label", "second_label", "confidence", "flag", "note"]
KEY_COLS = ["id", "batch", "mapped_label", "model_pred", "model_conf",
            "brand", "smishtank_category", "needs_sourcing"]


def load_rows():
    csv.field_size_limit(10 ** 9)
    with open(DATASET, encoding="utf-8", errors="replace") as fh:
        return [r for r in csv.DictReader(fh) if r["source"] == "SmishTank"]


def load_model_preds():
    """Stage-2 predictions, joined by text -- the eval csv carries no id."""
    if not os.path.exists(EVAL):
        return {}
    with open(EVAL, encoding="utf-8", errors="replace") as fh:
        return {r["text"]: (r["pred"], r.get("conf", "")) for r in csv.DictReader(fh)}


def build_pools(rows):
    """Split the corpus into one pool per bucket, in a fixed order."""
    pools = {t: [] for t in SCAM_TYPES}
    pools["__fallback__"] = []
    for r in rows:
        cat = r["category"].strip().lower()
        # The brand fallback in scam_types.py:59 -- these say `other` but almost
        # none of them are.
        if cat in BRAND_DECIDES and r["scam_type"] == OTHER:
            pools["__fallback__"].append(r)
        elif r["scam_type"] in pools:
            pools[r["scam_type"]].append(r)
    for p in pools.values():
        p.sort(key=lambda r: int(r["id"]) if r["id"].isdigit() else 0)
    return pools


def placeholder(scam_type, i):
    slug = scam_type.replace(" ", "-").lower()
    return {
        "id": f"SOURCE-{slug}-{i:02d}",
        "text": "",
        "_mapped": scam_type,
        "_brand": "",
        "_cat": "",
        "_needs_sourcing": "1",
    }


def take(pools, bucket, n, rng):
    """n rows for a bucket, padding with sourcing work-orders when the corpus is short."""
    pool = list(pools.get(bucket, []))
    rng.shuffle(pool)
    picked = [{
        "id": r["id"],
        "text": r["text"],
        "_mapped": r["scam_type"],
        "_brand": r["brand"],
        "_cat": r["category"],
        "_needs_sourcing": "0",
    } for r in pool[:n]]
    # family emergency, charity, Medicare and health have no rows at all and utility
    # shutoff has one, so the rest are blanks for you to go and find. They cannot be
    # blind-labelled in this pass -- see README.md.
    target = bucket if bucket != "__fallback__" else OTHER
    for i in range(len(picked), n):
        picked.append(placeholder(target, i + 1))
    return picked


def write_blind(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=BLIND_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({"id": r["id"], "text": r["text"],
                        "label": "", "second_label": "", "confidence": "",
                        "flag": "NEEDS_SOURCING" if r["_needs_sourcing"] == "1" else "",
                        "note": ""})


def write_key(path, rows, preds):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=KEY_COLS)
        w.writeheader()
        for r in rows:
            pred, conf = preds.get(r["text"], ("", "")) if r["text"] else ("", "")
            w.writerow({
                "id": r["id"],
                "batch": r["_batch"],
                "mapped_label": r["_mapped"],
                "model_pred": pred,
                "model_conf": conf,
                "brand": r["_brand"],
                "smishtank_category": r["_cat"],
                "needs_sourcing": r["_needs_sourcing"],
            })


def main():
    rng = random.Random(SEED)
    rows = load_rows()
    pools = build_pools(rows)

    full = []
    for bucket, n, _note in PLAN_340:
        picked = take(pools, bucket, n, rng)
        for r in picked:
            r["_batch"] = bucket
        full.extend(picked)

    # The 240 set is a strict subset: take the first k of each bucket, so ids match.
    by_bucket = {}
    for r in full:
        by_bucket.setdefault(r["_batch"], []).append(r)
    short_ids = set()
    for bucket, k in PLAN_240.items():
        short_ids.update(r["id"] for r in by_bucket[bucket][:k])
    short = [r for r in full if r["id"] in short_ids]

    assert len(full) == 340, len(full)
    assert len(short) == 240, len(short)
    assert short_ids <= {r["id"] for r in full}

    # Interleave everything. Labelling 180 delivery messages in a row turns into
    # pattern-matching on position instead of content.
    rng.shuffle(full)
    rng.shuffle(short)

    preds = load_model_preds()
    write_blind(os.path.join(HERE, "dataset_340.csv"), full)
    write_blind(os.path.join(HERE, "dataset_240.csv"), short)
    write_key(os.path.join(HERE, "answer_key.csv"), full, preds)

    src = sum(1 for r in full if r["_needs_sourcing"] == "1")
    print(f"dataset_340.csv  {len(full)} rows  ({len(full) - src} real, {src} to source)")
    src = sum(1 for r in short if r["_needs_sourcing"] == "1")
    print(f"dataset_240.csv  {len(short)} rows  ({len(short) - src} real, {src} to source)")
    print(f"answer_key.csv   {len(full)} rows  -- do not open before labelling")


if __name__ == "__main__":
    main()
