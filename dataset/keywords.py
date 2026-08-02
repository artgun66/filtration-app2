"""The arm-1 keyword rule, loaded from keywords.yaml.

run_arms.py scores arm 1 with this list, and merge_datasets.py writes one 0/1 column
per term into the dataset, so the columns and the arm cannot drift apart. Matching is
a literal lowercase substring test -- see keywords.yaml for why.
"""
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "keywords.yaml"), encoding="utf-8") as fh:
    KEYWORDS = yaml.safe_load(fh)["keywords"]


def column(kw):
    """Keyword -> CSV column name: 'gift card' -> 'kw_gift_card', 'www.' -> 'kw_www'."""
    return "kw_" + re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", kw.lower())).strip("_")


COLUMNS = [column(k) for k in KEYWORDS]
assert len(set(COLUMNS)) == len(COLUMNS), "two keywords collapsed to one column name"


def flags(text):
    """{column: 0|1} for one message, on the same substring test arm 1 uses."""
    low = (text or "").lower()
    return {c: (1 if k in low else 0) for k, c in zip(KEYWORDS, COLUMNS)}
