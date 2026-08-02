"""The plan's 29 engineered features: links, money, pressure, credentials, brands,
opt-out language, text shape and relational openers.

Every feature comes from the message body alone, so it works on the UCI and NUS rows
that ship no sender field. sender_type is deliberately excluded: only 955 rows have
one and they all come from SmishTank, so a model given it would learn provenance
rather than smishing. It belongs in the app as a rule.

The word lists live in features.yaml; the logic is here. No pandas above `build()`,
so merge_datasets.py can import this without it.
"""
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "features.yaml"), encoding="utf-8") as fh:
    _CFG = yaml.safe_load(fh)

# ---------------------------------------------------------------- links
# Shared with merge_datasets.py, which uses it for the has_url column.
URL_RE = re.compile(
    r"(https?://|www\.|\b[a-z0-9-]+\.(com|net|org|info|xyz|ru|cn|co|io|me|us|biz|top|"
    r"link|click|vip|shop|site|online|icu|cc|tk|ml|ga|gq|buzz)\b)", re.I)

# Full URL match, used where the domain itself has to be inspected. Group 1 is the
# scheme/www prefix (may be absent), group 2 the host.
URL_FULL_RE = re.compile(
    r"(https?://|www\.)?([a-z0-9][a-z0-9.-]*\.[a-z]{2,})(/\S*)?", re.I)

IP_DOMAIN_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
LEET_RE = re.compile(r"\b[a-z]+[0-9]+[a-z]+\b", re.I)   # M3dicare, Amaz0n
WORD_RE = re.compile(r"\b\w+\b")

# ---------------------------------------------------------------- vocabulary
KNOWN_TLDS = set(_CFG["known_tlds"])
SUSPICIOUS_TLDS = set(_CFG["suspicious_tlds"])
SHORTENERS = set(_CFG["shorteners"])
MISMATCH_BRANDS = _CFG["mismatch_brands"]
# Each family becomes two features: <family>_n (count) and <family>_hit (0/1).
FAMILIES = _CFG["families"]


def _domains(text):
    out = []
    for m in URL_FULL_RE.finditer(text):
        prefix, host = m.group(1), m.group(2).lower().rstrip(".")
        if "." not in host or host.replace(".", "").isdigit():
            continue                                   # "3.50" is not a host
        if not prefix and host.rsplit(".", 1)[-1] not in KNOWN_TLDS:
            continue                                   # "tomorrow.We" is not a host
        out.append(host)
    return out


def _registrable(host):
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def extract_features(text):
    """Return the engineered feature dict for one message body."""
    t = text or ""
    low = t.lower()
    flat = re.sub(r"[^a-z0-9]", "", low)
    n = len(t) or 1
    words = WORD_RE.findall(t)
    domains = _domains(t)
    tlds = {d.rsplit(".", 1)[-1] for d in domains}

    f = {}

    # --- links
    f["has_url"] = int(bool(URL_RE.search(t)))
    f["n_urls"] = len(domains)
    f["has_shortener"] = int(any(_registrable(d) in SHORTENERS for d in domains))
    f["suspicious_tld"] = int(bool(tlds & SUSPICIOUS_TLDS))
    f["ip_as_domain"] = int(bool(IP_DOMAIN_RE.search(t)) and bool(domains))
    f["url_digit_ratio"] = (
        sum(c.isdigit() for d in domains for c in d) / sum(len(d) for d in domains)
        if domains else 0.0)
    f["domain_len_max"] = max((len(d) for d in domains), default=0)
    # brand named in the body but absent from every link in it
    named = [b for b in MISMATCH_BRANDS if b in flat]
    f["brand_domain_mismatch"] = int(
        bool(named) and bool(domains)
        and not any(b in d.replace(".", "").replace("-", "") for b in named for d in domains))

    # --- keyword families
    for fam, terms in FAMILIES.items():
        hits = sum(low.count(term) for term in terms)
        f[f"{fam}_n"] = hits
        f[f"{fam}_hit"] = int(hits > 0)

    # --- text shape
    letters = sum(c.isalpha() for c in t)
    f["length"] = len(t)
    f["n_words"] = len(words)
    f["capital_ratio"] = sum(c.isupper() for c in t) / (letters or 1)
    f["digit_ratio"] = sum(c.isdigit() for c in t) / n
    f["punct_ratio"] = sum(not c.isalnum() and not c.isspace() for c in t) / n
    f["n_exclaim"] = t.count("!")
    f["n_allcaps_words"] = sum(1 for w in words if len(w) > 2 and w.isupper())
    f["has_leetspeak"] = int(bool(LEET_RE.search(t)))
    f["avg_word_len"] = (sum(len(w) for w in words) / len(words)) if words else 0.0

    return f


# Order is fixed by a call on a probe string so downstream arrays stay aligned.
FEATURE_NAMES = list(extract_features("probe http://a.com").keys())

# Readable labels for SHAP plots (plan Phase 6).
FEATURE_LABELS = {
    "has_url": "Contains a link",
    "n_urls": "Number of links",
    "has_shortener": "Link uses a URL shortener",
    "suspicious_tld": "Suspicious top-level domain",
    "ip_as_domain": "IP address used as a domain",
    "url_digit_ratio": "Digits in domain (ratio)",
    "domain_len_max": "Longest domain length",
    "brand_domain_mismatch": "Brand named but link does not match",
    "money_n": "Money terms (count)",
    "money_hit": "Mentions money transfer",
    "pressure_n": "Urgency terms (count)",
    "pressure_hit": "Uses urgency / pressure",
    "credentials_n": "Credential terms (count)",
    "credentials_hit": "Asks for credentials",
    "brands_n": "Brand mentions (count)",
    "brands_hit": "Mentions a known brand",
    "optout_n": "Opt-out phrases (count)",
    "optout_hit": "Carries legal opt-out language",
    "opener_n": "Relational openers (count)",
    "opener_hit": "Uses a relational opener",
    "length": "Message length",
    "n_words": "Word count",
    "capital_ratio": "Capital-letter ratio",
    "digit_ratio": "Digit ratio",
    "punct_ratio": "Punctuation ratio",
    "n_exclaim": "Exclamation marks",
    "n_allcaps_words": "ALL-CAPS words",
    "has_leetspeak": "Character substitution (M3dicare)",
    "avg_word_len": "Average word length",
}


# The features are a property of the dataset, but the parquet is a training input,
# so it is written next to the models that read it.
DATASET_CSV = os.path.join(HERE, "combined_sms_dataset.csv")
PARQUET = os.path.join(HERE, os.pardir, "scam-classification", "features.parquet")


def build(src=DATASET_CSV, out=PARQUET):
    import pandas as pd

    df = pd.read_csv(src, keep_default_na=False, dtype={"text": str})
    feats = pd.DataFrame([extract_features(t) for t in df["text"]], columns=FEATURE_NAMES)
    feats.insert(0, "id", df["id"].values)
    path = os.path.abspath(out)
    feats.to_parquet(path, index=False)
    return df, feats, path


if __name__ == "__main__":
    df, feats, path = build()
    print(f"{len(feats):,} rows x {len(FEATURE_NAMES)} features -> {path}\n")

    # Sanity view: mean of each feature per label. A feature that does not separate
    # smishing from ham here is unlikely to earn its place in the model.
    view = feats.drop(columns=["id"]).groupby(df["label"].values).mean().T
    view = view[[c for c in ("ham", "spam", "smishing") if c in view.columns]]
    print(view.round(3).to_string())
