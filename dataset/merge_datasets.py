"""Merge the five SMS / smishing datasets into one labeled CSV.

Unpacks each source into raw/ under its dataset name first, so the build is
reproducible from what is checked into the project. Writes combined_sms_dataset.csv
and a report; see combined_sms_dataset.md for what every column means.

  python merge_datasets.py
"""
import csv, io, json, re, collections, os, zipfile

import yaml

import features
import keywords
import scam_types

csv.field_size_limit(10 ** 9)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")            # unpacked, named after the dataset
ARCHIVES = os.path.join(HERE, "archives")  # the files as downloaded

OUT = os.path.join(HERE, "combined_sms_dataset.csv")
REPORT = os.path.join(HERE, "combined_sms_dataset_report.txt")

with open(os.path.join(HERE, "sources.yaml"), encoding="utf-8") as fh:
    _CFG = yaml.safe_load(fh)
SOURCES = _CFG["sources"]

# The 29 engineered features, prefixed so feat_has_url does not collide with the
# has_url column above. Same values as features.parquet, floats rounded for the CSV.
FEAT_COLUMNS = ["feat_" + n for n in features.FEATURE_NAMES]

FIELDS = [
    # schema the analysis plan asks for
    "id", "text", "label", "scam_type", "source", "source_link", "date",
    # provenance and split control
    "spam_label", "smishing_label", "scam", "is_clean_label", "dup_group",
    "source_id", "has_url", "label_conflict", "dup_count", "also_in",
    # enrichment, SmishTank only
    "sender", "sender_type", "brand", "category",
    "url", "domain", "tld", "domain_registrar",
    # written blank here and filled by annotate.py, which needs the trained model.
    # They live in the schema so the column order does not depend on run order.
    "split", "model_pred",
    # one 0/1 column per term in the arm-1 keyword rule, kw_http ... kw_sign_in
] + keywords.COLUMNS + FEAT_COLUMNS

# Source names, so the loaders below can refer to one without retyping the string.
SMISHTANK = "SmishTank"
SMISHING_DS = "Smishing_Dataset"
NUS = "NUS SMS Corpus"
MENDELEY = "Mendeley SMS Phishing"
UCI = "UCI SMS Spam Collection"
assert set(SOURCES) == {SMISHTANK, SMISHING_DS, NUS, MENDELEY, UCI}, \
    "sources.yaml does not list the five sources this script loads"

SOURCE_LINK = {s: c["link"] for s, c in SOURCES.items()}

# Whose label wins when the same text appears in several sources, and whose
# enrichment columns are carried. They differ on purpose -- see sources.yaml.
LABEL_PRIORITY = {s: c["label_priority"] for s, c in SOURCES.items()}
FIELD_PRIORITY = {s: c["field_priority"] for s, c in SOURCES.items()}

# Sources whose labels were machine-assigned. is_clean_label drives evaluation:
# val/test are drawn from clean rows only.
LEAKY_SOURCES = {s for s, c in SOURCES.items() if c.get("leaky")}

URL_RE = features.URL_RE          # one definition, shared with the feature builder
WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------- unpack sources
def _write(source, data):
    path = os.path.join(RAW, SOURCES[source]["raw"])
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _read_source(spec):
    """Pull one source's bytes out of archives/, however it happens to be packaged."""
    kind = spec["type"]
    if kind == "file":
        return open(os.path.join(ARCHIVES, spec["path"]), "rb").read()
    with zipfile.ZipFile(os.path.join(ARCHIVES, spec["archive"])) as z:
        if kind == "zip":
            return z.read(spec["member"])
        if kind == "nested_zip":
            with zipfile.ZipFile(io.BytesIO(z.read(spec["member"]))) as z2:
                return z2.read(spec["inner_member"])
    raise ValueError(f"unknown source type {kind!r}")


def extract_sources():
    """Unpack every source into raw/, keyed by source name. Idempotent."""
    return {s: _write(s, _read_source(c["from"])) for s, c in SOURCES.items()}
    return paths


PATHS = extract_sources()


def read_text(path):
    b = open(path, "rb").read()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("latin-1")


def rows_of(path):
    """CSV rows as dicts. StringIO(newline='') keeps csv in charge of record
    splitting -- str.splitlines() would also break on \\x85 / \\u2028 and shred rows."""
    return csv.DictReader(io.StringIO(read_text(path), newline=""))


def clean(s):
    """Collapse embedded newlines/whitespace; strip control chars."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\u00a0", " ")
    s = "".join(ch for ch in s if ch == "\t" or ch >= " " or ch in "\r\n")
    return WS_RE.sub(" ", s).strip()


def key(msg):
    return WS_RE.sub(" ", msg.lower()).strip()


_TS_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")


def iso_date(s):
    """SmishTank ships 'MM/DD/YYYY, HH:MM:SS'. Keep the date only -- it is the
    one source with a timestamp, and the plan's schema asks for a date column."""
    m = _TS_RE.match(s or "")
    return f"{m.group(3)}-{m.group(1)}-{m.group(2)}" if m else ""


# Smishing is templated: the same scam is resent with a fresh URL and a new amount,
# which exact-text dedupe keeps as distinct rows. Blanking the parts that vary gives
# a template key; rows sharing one form a dup_group, and splits are made on the group.
_URL_ANY = re.compile(r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.[a-z]{2,}(/\S*)?)", re.I)
_DIGITS = re.compile(r"\d+")
_NON_WORD = re.compile(r"[^a-z ]+")
# Placeholders must be lowercase words: _NON_WORD would otherwise erase them and
# every link-only message would collapse into a single group.
_PLACEHOLDER = re.compile(r"\b(urltoken|numtoken)\b")


def template_key(msg):
    s = msg.lower()
    s = _URL_ANY.sub(" urltoken ", s)          # before digits: URLs contain digits
    s = _DIGITS.sub(" numtoken ", s)
    s = _NON_WORD.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    # A message that was nothing but a link, a number or punctuation leaves no words
    # behind. ":)" and "." are not the same template any more than two unrelated bare
    # URLs are, so fall back to the exact text and let them stand alone.
    bare = _PLACEHOLDER.sub("", s).strip()
    if not bare:
        return "=" + key(msg)
    return s


rows = []
counts = collections.Counter()


def add(source, source_id, message, spam, smishing, **extra):
    message = clean(message)
    if not message:
        counts[source + ":dropped_empty"] += 1
        return
    r = {f: "" for f in FIELDS}
    r.update(extra)
    r["source"] = source
    r["source_link"] = SOURCE_LINK[source]
    r["source_id"] = source_id
    r["text"] = message
    r["spam_label"] = "" if spam is None else spam
    r["smishing_label"] = "" if smishing is None else smishing
    r["is_clean_label"] = 0 if source in LEAKY_SOURCES else 1
    has_url = extra.get("has_url")
    r["has_url"] = has_url if has_url != "" and has_url is not None else (1 if URL_RE.search(message) else 0)
    rows.append(r)
    counts[source] += 1


# ---------------------------------------------------------------- 1. SmishTank
# Curated smishing screenshots with URL / WHOIS / brand enrichment. Every row is smishing.
p = PATHS[SMISHTANK]
for x in rows_of(p):
    text = clean(x.get("MainText")) or clean(x.get("Fulltext"))
    cat, brand = clean(x.get("Message Categories")), clean(x.get("Brand"))
    add(SMISHTANK, clean(x.get("messageid")), text, 1, 1,
        sender=clean(x.get("Sender")), sender_type=clean(x.get("SenderType")),
        brand=brand, category=cat,
        scam_type=scam_types.scam_type_for(cat, brand),
        date=iso_date(clean(x.get("timeReceived"))),
        url=clean(x.get("Url")), domain=clean(x.get("Domain")),
        tld=clean(x.get("TLD")), domain_registrar=clean(x.get("Domain Registrar")),
        has_url=1 if clean(x.get("Url")) else "")

# ---------------------------------------------------------------- 2. Smishing_Dataset
# Not clean: 624 rows carry the literal "Smishing" in the spam-label column (a
# leaked class name from the Mendeley set) and 1,076 have no spam label at all.
p = PATHS[SMISHING_DS]
for i, x in enumerate(rows_of(p), 1):
    sp, sm = clean(x.get("spam label")), clean(x.get("smishing label"))
    smish = int(sm) if sm in ("0", "1") else None
    if sp in ("0", "1"):
        spam = int(sp)
    elif sp.lower() == "smishing":
        spam, smish = 1, 1
    elif sp == "":
        spam = 1 if smish == 1 else None      # unlabelled ham/spam, keep as unknown
    else:
        counts[SMISHING_DS + ":dropped_badlabel"] += 1
        continue
    add(SMISHING_DS, i, x.get("message"), spam, smish)

# ---------------------------------------------------------------- 3. NUS SMS Corpus (English) — legitimate messages
p = PATHS[NUS]
data = json.load(open(p, encoding="utf-8"))
for m in data["smsCorpus"]["message"]:
    t = m.get("text", {})
    add(NUS, m.get("@id", ""), t.get("$") if isinstance(t, dict) else t, 0, 0)

# ---------------------------------------------------------------- 4. Mendeley SMS Phishing
# Three classes, not two: ham / spam / Smishing (case varies).
LAB_MENDELEY = {k: tuple(v) for k, v in _CFG["label_mapping"]["mendeley"].items()}
p = PATHS[MENDELEY]
for i, x in enumerate(rows_of(p), 1):
    lab = LAB_MENDELEY.get(clean(x.get("LABEL")).lower())
    if lab is None:
        counts[MENDELEY + ":dropped_badlabel"] += 1
        continue
    add(MENDELEY, i, x.get("TEXT"), lab[0], lab[1],
        has_url=1 if clean(x.get("URL")).lower() == "yes" else 0)

# ---------------------------------------------------------------- 5. UCI SMS Spam Collection (tab separated)
p = PATHS[UCI]
for i, line in enumerate(read_text(p).splitlines(), 1):
    if not line.strip():
        continue
    lab, _, text = line.partition("\t")
    lab = lab.strip().lower()
    if lab not in ("ham", "spam"):
        counts[UCI + ":dropped_badlabel"] += 1
        continue
    # ham implies not-smishing; spam leaves smishing unannotated (as in Mendeley)
    add(UCI, i, text, *((1, None) if lab == "spam" else (0, 0)))

raw_total = len(rows)

# ---------------------------------------------------------------- dedupe
NO_LABEL = 99          # sorts below every real LABEL_PRIORITY

best = {}
order = []
for r in rows:
    k = key(r["text"])
    pri = LABEL_PRIORITY[r["source"]]
    if k not in best:
        best[k] = r
        r["dup_count"] = 1
        r["_srcs"] = {r["source"]}
        r["_spams"] = {r["spam_label"]} - {""}
        # Priority tracked per field, so a Mendeley "spam" with smishing unannotated
        # can still take its smishing flag from a lower-priority source.
        r["_sp_pri"] = pri if r["spam_label"] != "" else NO_LABEL
        r["_sm_pri"] = pri if r["smishing_label"] != "" else NO_LABEL
        order.append(k)
        continue
    w = best[k]
    w["dup_count"] += 1
    w["_srcs"].add(r["source"])
    if r["spam_label"] != "":
        w["_spams"].add(r["spam_label"])
        if pri < w["_sp_pri"]:
            w["spam_label"], w["_sp_pri"] = r["spam_label"], pri
    if r["smishing_label"] != "" and pri < w["_sm_pri"]:
        w["smishing_label"], w["_sm_pri"] = r["smishing_label"], pri
    # Enrichment columns follow the richest source, independently of the label --
    # SmishTank carries the WHOIS/brand data even where it does not win the label.
    if FIELD_PRIORITY[r["source"]] < FIELD_PRIORITY[w["source"]]:
        keep = {f: r[f] for f in FIELDS if f not in
                ("id", "dup_group", "dup_count", "also_in", "label",
                 "spam_label", "smishing_label", "scam", "is_clean_label",
                 "split", "model_pred")}
        w.update(keep)
        w["source"] = r["source"]

# ---------------------------------------------------------------- resolve labels
# Leaving these blank pushes the decision onto whoever loads the CSV, so settle them
# here and count each resolution in the report.
for k in order:
    r = best[k]
    # smishing implies suspicious: 'ham' + smishing=1 is a contradiction, not a third class
    if r["spam_label"] == 0 and r["smishing_label"] == 1:
        r["spam_label"] = 1
        counts["resolved:ham_but_smishing"] += 1
    # UCI and Mendeley publish 'spam' without saying whether it is smishing. Per
    # the plan's label mapping those are the advertisement class, so: not smishing.
    if r["spam_label"] == 1 and r["smishing_label"] == "":
        r["smishing_label"] = 0
        counts["resolved:spam_smishing_unknown"] += 1
    if r["spam_label"] == "":
        r["spam_label"] = 1 if r["smishing_label"] == 1 else 0
        counts["resolved:spam_unknown"] += 1

    r["label"] = ("smishing" if r["smishing_label"] == 1
                  else "spam" if r["spam_label"] == 1 else "ham")
    # Until here spam_label carries the sources' own meaning, "not ham" -- what the
    # priority resolution above needs to compare. Now that `label` is settled, recode
    # to the exclusive form: scam = spam or smishing, spam_label = spam only.
    r["scam"] = 1 if r["label"] in ("spam", "smishing") else 0
    r["spam_label"] = 1 if r["label"] == "spam" else 0
    # a clean source having seen the same text is what makes the label trustworthy
    r["is_clean_label"] = 0 if r["_srcs"] <= LEAKY_SOURCES else 1

# ---------------------------------------------------------------- near-duplicate groups
groups = {}
for k in order:
    r = best[k]
    r["dup_group"] = groups.setdefault(template_key(r["text"]), len(groups) + 1)

out = []
for i, k in enumerate(order, 1):
    r = best[k]
    r["id"] = i
    r["label_conflict"] = 1 if len(r["_spams"]) > 1 else 0
    r["also_in"] = "|".join(sorted(s for s in r["_srcs"] if s != r["source"]))
    r.update(keywords.flags(r["text"]))
    for name, v in features.extract_features(r["text"]).items():
        r["feat_" + name] = round(v, 6) if isinstance(v, float) else v
    out.append({f: r[f] for f in FIELDS})

with open(OUT, "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(out)

# ---------------------------------------------------------------- report
kept = collections.Counter(r["source"] for r in out)
lines = []
lines.append("Combined SMS / smishing dataset")
lines.append("=" * 60)
lines.append("")
lines.append("Rows read per source (after dropping blanks / bad labels):")
for s in FIELD_PRIORITY:
    lines.append(f"  {s:<24} {counts[s]:>7,}   kept after dedupe: {kept[s]:>6,}")
    lines.append(f"    {SOURCE_LINK[s]}")
for k2 in sorted(counts):
    if ":" in k2:
        lines.append(f"  ! {k2}: {counts[k2]}")
lines.append("")
lines.append(f"Total rows read      : {raw_total:,}")
lines.append(f"Exact-text duplicates: {raw_total - len(out):,}")
lines.append(f"Final unique rows    : {len(out):,}")
lines.append("")
sp = collections.Counter(r["spam_label"] for r in out)
sm = collections.Counter(r["smishing_label"] for r in out)
sc = collections.Counter(r["scam"] for r in out)
lines.append(f"scam            1 (spam or smishing) : {sc[1]:>7,}")
lines.append(f"scam            0 (ham)              : {sc[0]:>7,}")
lines.append(f"spam_label      1 (spam, not smish)  : {sp[1]:>7,}")
lines.append(f"smishing_label  1 (smishing)         : {sm[1]:>7,}")
# the three classes are mutually exclusive, so the two halves must add to the target
lines.append(f"  scam == spam_label + smishing_label: "
             f"{all(r['scam'] == r['spam_label'] + r['smishing_label'] for r in out)}")
lines.append("")
lab = collections.Counter(r["label"] for r in out)
for name in ("ham", "spam", "smishing"):
    lines.append(f"label  {name:<10}          : {lab[name]:>7,}")
lines.append("")
clean_n = sum(1 for r in out if r["is_clean_label"] == 1)
lines.append(f"Clean-labelled rows      : {clean_n:>7,}   (val/test are drawn from these)")
lines.append(f"Keyword-labelled rows    : {len(out) - clean_n:>7,}   (train only)")
lines.append("")
ng = len({r["dup_group"] for r in out})
biggest = collections.Counter(r["dup_group"] for r in out).most_common(1)[0][1]
lines.append(f"Near-duplicate groups    : {ng:>7,}   (split on these, not on rows)")
lines.append(f"  rows collapsed by them : {len(out) - ng:>7,}")
lines.append(f"  largest group          : {biggest:>7,} rows")
lines.append("")
st = collections.Counter(r["scam_type"] for r in out if r["scam_type"])
lines.append(f"Rows with a scam_type    : {sum(st.values()):>7,}")
for t in scam_types.SCAM_TYPES:
    if st[t]:
        lines.append(f"    {t:<26} {st[t]:>5,}")
missing = [t for t in scam_types.SCAM_TYPES if not st[t]]
if missing:
    lines.append(f"  no examples yet: {', '.join(missing)}")
lines.append("")
lines.append(f"Rows with a URL          : {sum(1 for r in out if r['has_url'] == 1):,}")
lines.append(f"Rows with a date         : {sum(1 for r in out if r['date']):,}")
lines.append(f"Cross-source duplicates  : {sum(1 for r in out if r['also_in']):,}")
lines.append(f"Conflicting scam labels  : {sum(1 for r in out if r['label_conflict'] == 1):,}")
lines.append("")
lines.append(f"Keyword columns ({len(keywords.COLUMNS)}), rows matched and how they split by label:")
lines.append(f"  {'column':<20}{'rows':>8}{'ham':>8}{'spam':>7}{'smish':>7}   lift")
for c in keywords.COLUMNS:
    hit = [r for r in out if r[c] == 1]
    if not hit:
        lines.append(f"  {c:<20}{0:>8}")
        continue
    n = len(hit)
    by = collections.Counter(r["label"] for r in hit)
    # P(suspicious | keyword present) / P(suspicious) -- 1.0 means the term carries
    # no signal at all, below 1.0 means it points the wrong way
    lift = (n - by["ham"]) / n / (sum(1 for r in out if r["label"] != "ham") / len(out))
    lines.append(f"  {c:<20}{n:>8,}{by['ham']:>8,}{by['spam']:>7,}{by['smishing']:>7,}"
                 f"{lift:>7.2f}")
txt = "\n".join(lines)
open(REPORT, "w", encoding="utf-8").write(txt + "\n")
print(txt)
