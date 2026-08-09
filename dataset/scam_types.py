"""Map SmishTank's curated categories onto the plan's 13 scam types.

The taxonomy, the category map and the brand lists live in scam_types.yaml; this
module is the lookup logic. `Message Categories` is the primary signal; `Brand` only
refines the buckets that are genuinely ambiguous (Account Alert, Other), where the
category says nothing about who is impersonated.

  python scam_types.py     # count how many rows each type gets
"""
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "scam_types.yaml"), encoding="utf-8") as fh:
    _CFG = yaml.safe_load(fh)

SCAM_TYPES = _CFG["scam_types"]
CATEGORY_MAP = _CFG["category_map"]
BRAND_DECIDES = set(_CFG["brand_decides"])
UNCOVERED = _CFG["uncovered"]
THIN = _CFG["thin"]

# Named constants, so the stage-2 prompt can refer to a type without retyping it.
(GOVERNMENT, TECH_SUPPORT, BANK, DELIVERY, FAMILY, ROMANCE, CRYPTO,
 PRIZE, CHARITY, HEALTH, UTILITY, JOB, OTHER, NOT_SCAM) = SCAM_TYPES

# NOT_SCAM is the answer for a stage-1 false positive, not a kind of scam, so nothing
# in this module ever returns it: no SmishTank category or brand maps to it, and its
# training rows are ham. It exists in the taxonomy because stage 2 only ever sees
# already-flagged messages and needs a way to say the flag was wrong.
# app-backend/distill.py builds the class; scam_type_for below cannot produce it.
SCAM_ONLY_TYPES = [t for t in SCAM_TYPES if t != NOT_SCAM]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(s):
    return _NON_ALNUM.sub("", (s or "").lower())


# Keys normalised on load, so the yaml can be written readably.
BRAND_TO_TYPE = {_norm(b): t for t, brands in _CFG["brands"].items() for b in brands}

# A typo in the yaml would otherwise silently produce a 14th type.
_targets = set(CATEGORY_MAP.values()) | set(_CFG["brands"]) | set(UNCOVERED) | set(THIN)
assert not _targets - set(SCAM_TYPES), \
    f"scam_types.yaml maps to types not in the taxonomy: {sorted(_targets - set(SCAM_TYPES))}"


def scam_type_for(category, brand):
    """Return one of SCAM_TYPES, or "" when the row is not a categorised scam."""
    cat = (category or "").strip().lower()
    if not cat:
        return ""
    if cat in CATEGORY_MAP:
        return CATEGORY_MAP[cat]
    if cat in BRAND_DECIDES:
        # A brand cell can list several ("CVS, PUBLIX, HOME DEPOT"); first hit wins.
        for part in re.split(r"[,/]", brand or ""):
            hit = BRAND_TO_TYPE.get(_norm(part))
            if hit:
                return hit
        return OTHER
    return OTHER


if __name__ == "__main__":
    import csv, io, collections

    csv.field_size_limit(10 ** 9)
    p = os.path.join(HERE, "archives", "analysisdataset.csv")
    text = open(p, encoding="utf-8", errors="replace").read()
    counts = collections.Counter()
    for row in csv.DictReader(io.StringIO(text, newline="")):
        counts[scam_type_for(row.get("Message Categories"), row.get("Brand"))] += 1

    print(f"{sum(counts.values()):,} rows mapped\n")
    for t in SCAM_ONLY_TYPES:
        flag = "   <- no examples, hand-label needed" if counts[t] == 0 else ""
        print(f"  {t:<28} {counts[t]:>5,}{flag}")
    # Not listed above: it has no SmishTank rows by construction, so a zero here would
    # read as a gap to hand-label rather than as the design.
    print(f"\n  {NOT_SCAM:<28} {'--':>5}   ham rows, built by app-backend/distill.py")
    if counts[""]:
        print(f"  {'(uncategorised)':<28} {counts['']:>5,}")
