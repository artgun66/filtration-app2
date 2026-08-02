# `combined_sms_dataset.csv` — column reference

88,542 unique SMS messages, merged from the five sources in the analysis plan and deduped
on exact text. Built by `merge_datasets.py`; the last two columns are filled afterwards by
`../scam-classification/annotate.py`. Row counts and totals in this file are from the current build — see
`combined_sms_dataset_report.txt` for the build's own summary.

Every column is always present. Blank means "not known for this row", never "zero" — the
merge writes empty strings, so load with `keep_default_na=False` to keep them as `""`
rather than `NaN`.

## At a glance

| column | type | filled | what it is |
|---|---|---|---|
| `id` | int | 100% | row number, 1–88,542. The key everything else joins on |
| `text` | str | 100% | the message, whitespace-collapsed and control-chars stripped |
| `label` | str | 100% | `ham` / `spam` / `smishing` |
| `scam_type` | str | 1.2% | one of the plan's 13 scam types |
| `source` | str | 100% | which dataset the row's *fields* came from |
| `source_link` | str | 100% | where that dataset was downloaded from |
| `date` | str | 1.2% | `YYYY-MM-DD` received date |
| `spam_label` | 0/1 | 100% | 1 = spam, and not smishing |
| `smishing_label` | 0/1 | 100% | 1 = smishing (SMS phishing) |
| `scam` | 0/1 | 100% | the stage-1 target: 1 if either of the above |
| `is_clean_label` | 0/1 | 100% | 0 = label came only from keyword matching |
| `dup_group` | int | 100% | near-duplicate template group, 1–83,473 |
| `source_id` | str | 100% | the row's id in its original dataset |
| `has_url` | 0/1 | 100% | message contains a link |
| `label_conflict` | 0/1 | 100% | sources disagreed on `spam_label` |
| `dup_count` | int | 100% | how many exact copies collapsed into this row |
| `also_in` | str | 42.9% | the other sources that had this text, `\|`-separated |
| `sender` | str | 1.1% | sending number / short code / address |
| `sender_type` | str | 1.1% | `Phone Number` / `Short Code` / `Email To Text` |
| `brand` | str | 0.8% | brand impersonated |
| `category` | str | 1.2% | source's own category, before mapping to `scam_type` |
| `url` | str | 1.1% | link extracted from the message |
| `domain` | str | 1.0% | its domain |
| `tld` | str | 1.0% | its TLD |
| `domain_registrar` | str | 1.0% | WHOIS registrar |
| `split` | str | 100% | `train` / `val` / `test` |
| `model_pred` | 0/1 | 100% | classifier's call at its operating threshold |
| `kw_http` … `kw_sign_in` | 0/1 | 100% | 38 columns, one per term in the keyword rule |
| `feat_has_url` … `feat_avg_word_len` | num | 100% | the 29 engineered features |

## The labels

Four columns, one decision. `label` is the human-readable class; the three binary columns
split it into the two halves and their union. They are mutually exclusive and exhaustive by
construction, so `scam == spam_label + smishing_label` on every row (the merge asserts it):

| `label` | `spam_label` | `smishing_label` | `scam` | rows | |
|---|---|---|---|---|---|
| `ham` | 0 | 0 | 0 | 58,889 | normal message |
| `spam` | **1** | 0 | 1 | 6,623 | suspicious, not annotated as phishing |
| `smishing` | 0 | **1** | 1 | 23,030 | SMS phishing |

**`scam` is the stage-1 target**: 1 if the message is spam or smishing, 0 if ham. The
plan's stage-1 question is *safe vs suspicious*, and that is exactly this column.
`../scam-classification/modeling.py` derives the same thing as `y = (label != "ham")`.

**`spam_label` and `smishing_label` are the two halves**, and only their union is
trustworthy. The boundary between them is not:

* Sources publish "spam" without saying whether it is phishing. The merge assigns those
  `smishing_label = 0` by default (`merge_datasets.py:342-344`), so anything unannotated
  falls into `spam` — it is a residual bucket, not a clean advertisement class.
* It shows in the data. The 172 clean-labelled spam rows are UK premium-rate SMS from
  2011–12, and several read as outright scams — *"Congratulations ur awarded either £500 of
  CD gift vouchers"* is prize-and-lottery bait sitting under `spam` only because UCI ships
  two classes. The other 6,451 are keyword-labelled Indian commercial SMS, mostly genuine
  retail promos but including an advance-fee scam (*"Nigeria Mega Jackpot"*) and a
  link lure (*"You've been selected for a special discount … http://…"*).

Train on `scam`. Treat `spam_label` as "suspicious, and nobody said it was phishing" rather
than as a clean ad label. It costs little in evaluation terms either way: 6,451 of the
6,623 are `is_clean_label = 0`, so they never leave the training split, and test holds 13
spam rows against 341 smishing.

**A note on where these come from.** `spam_label` carries a different meaning inside the
merge than it does in the CSV. Smishing_Dataset publishes `spam label = 1` on its 22,637
smishing rows too — there it means "not ham" — and the merge keeps that convention while it
resolves conflicting sources by priority, because that is the comparison it needs. Once
`label` is settled the columns are recoded to the exclusive form above
(`merge_datasets.py:350-360`).

**`scam_type`** subdivides the smishing rows into the plan's 13 types, and only SmishTank
supplies it — 1,055 rows:

```
other 360 · delivery and toll 180 · bank alert 116 · tech support 102
government impersonation 85 · romance 65 · investment and crypto 64
prize and lottery 57 · job offer 25 · utility shutoff 1
```

Three types have no examples at all — family emergency, charity, Medicare and health — and
utility shutoff has one. Stage 2 cannot be scored on those until they're hand-labelled.
`category` is SmishTank's own raw category on the same rows; `scam_types.py` maps it to
`scam_type`.

## Provenance

**`source` names who supplied the row's fields, which is not always who supplied its
label.** When the same text appears in several datasets the merge keeps one row, and it
resolves the two questions with separate rankings:

* `LABEL_PRIORITY` — whose label to trust: SmishTank → Mendeley → UCI → NUS →
  Smishing_Dataset (last, because its labels were keyword-assigned).
* `FIELD_PRIORITY` — whose enrichment columns to carry: SmishTank → Smishing_Dataset →
  Mendeley → UCI → NUS.

So a row can read `source = Smishing_Dataset` while its `ham` verdict actually came from
NUS. `also_in` lists the other sources that had the same text, so the full provenance is
recoverable: 37,996 rows appear in more than one source.

**`source_id`** is the row's identifier in that original file, for going back to it.

**`is_clean_label`** is 0 when the *only* source for a text is Smishing_Dataset, whose
labels were assigned by keyword matching. 36,484 rows are keyword-labelled; the other
52,058 have at least one hand-checked source. This column is load-bearing: keyword rows
may train, but they never enter validation or test, because scoring against them would
measure agreement with a keyword rule rather than detection of smishing.

**`label_conflict`** is 1 when sources disagreed on whether a text is suspicious at all
(that is, on `scam`, though it is computed on the pre-recode `spam_label`) — 85
rows, mostly ad-like messages that NUS publishes as ham and Smishing_Dataset's keyword rule
flagged as spam. The kept label is the higher-priority source's. Diagnostic only; nothing
in the pipeline reads it.

## Duplicates

Two different notions, both needed:

**`dup_count`** — how many rows with byte-identical text collapsed into this one. 1 for a
unique message, up to 230. 153,305 rows were read and 64,763 exact duplicates removed.

**`dup_group`** — near-duplicate group. Smishing is templated: the same scam is resent with
a fresh tracking URL and a new order number, which survives exact-text dedupe. Blanking the
parts that vary gives a template key, and rows sharing one get the same `dup_group`
(83,473 groups, largest holds 28 rows).

**Splits are made on `dup_group`, never on the row.** A row-level split would put
near-identical messages on both sides of it and inflate every score.

## Enrichment (SmishTank rows only)

`sender`, `sender_type`, `brand`, `category`, `url`, `domain`, `tld`, `domain_registrar`
come from SmishTank's URL/WHOIS analysis and are blank on ~99% of rows. They are useful for
analysis and for app-side rules, but note:

* **`sender_type` is deliberately not a model feature.** Only 955 rows have one and they
  all come from one source, so a model given it would learn provenance, not smishing.
* `date` has the same shape — only SmishTank timestamps its messages, so the 1,055 dated
  rows run 2022-03-31 to 2023-12-13 while the rest of the corpus is undated.

`has_url` is the exception: it is filled for every row, taken from the source where the
source states it and otherwise detected from the text by regex. 8,758 rows have a link —
7,179 of the 23,030 smishing messages, against 286 of 58,889 ham.

## Split and prediction

Both are written by `../scam-classification/annotate.py`, not by the merge. Re-running `merge_datasets.py` blanks
them; re-run `../scam-classification/annotate.py` to refill.

**`split`** — `train` (68,302) / `val` (10,120) / `test` (10,120), from
`../scam-classification/modeling.py`'s `make_splits`. Deterministic at `SEED = 200`, so it is the same split
`../scam-classification/run_arms.py` scored every arm against. Two invariants hold and are asserted, not assumed:
no `dup_group` straddles a split, and val/test contain only `is_clean_label = 1` rows. The
consequence is a deliberately lopsided test set — 354 positives at a 3.5% positive rate,
against 42% in train — because the clean sources are overwhelmingly NUS ham.

**`model_pred`** — hard 0/1 from the `qwen_feat_clean` arm at 0.461, the threshold fitted
on validation to hold 95% recall. Two caveats:

* It is filled for **every** row, including ones the model trained on, where it is
  in-sample and far too flattering. Score it on `split == "test"` only, where it gives
  recall 0.929, specificity 0.998, precision 0.951.
* On train it flags 15,265 of 28,945 positives. That is not a failure — this arm is the
  ablation that never trained on the 36,484 keyword-labelled rows, and it disagrees with a
  lot of them. Which, per the project's findings, is the point.

## Keyword columns

38 columns, `kw_http` through `kw_sign_in`, one per term in the keyword rule that
`../scam-classification/run_arms.py` scores as arm 1. 1 if the term appears in `text`, 0 if not. The list lives in
`keywords.yaml` so the columns and the arm cannot drift apart — `sum(kw_*) / 38` reproduces
arm 1's score exactly, which is asserted rather than assumed.

Column names slugify the term: `gift card` → `kw_gift_card`, `www.` → `kw_www`,
`update your` → `kw_update_your`.

**Matching is a literal lowercase substring test**, because that is what the arm does. It is
not word-boundary aware, and the report's per-column breakdown shows what that costs:

```
column                  rows     ham   spam  smish   lift
kw_http                9,871      46  1,813  8,012   2.97
kw_irs                 1,741   1,039    146    556   1.20
kw_won                 1,255     751     16    488   1.20
kw_social_security         2       1      0      1   1.49
```

`lift` is P(suspicious | term present) ÷ P(suspicious); 1.0 means the term carries no
signal. `kw_irs` fires on 1,039 ham messages because "irs" sits inside *first*, *girls*,
*thirsty*; `kw_won` likewise inside *town* and *wonder*. Both land at 1.20 — barely above
chance. At the other end `kw_http` reaches 2.97 against a ceiling of 2.99.

This is intentional. Arm 1 is the sanity floor and a stand-in for whatever rule produced
Smishing_Dataset's labels, so it is scored the way such a rule really behaves. If you want
word-boundary matching for feature engineering rather than for reproducing the arm, change
`keywords.py`'s `flags` to a `\b`-anchored regex — but then the columns no longer reproduce arm 1,
so change `keyword_score` with it.

64,496 of 88,542 rows match no keyword at all.

## Engineered feature columns

29 columns, `feat_has_url` through `feat_avg_word_len` — the features `qwen_feat` and
`qwen_feat_clean` train on, from `features.py` / `features.yaml`. Prefixed because `feat_has_url` would
otherwise collide with the dataset's own `has_url` (which prefers what the source states;
`feat_has_url` is always regex-detected). Floats are rounded to 6 decimals for the CSV.

Same values as `../scam-classification/features.parquet`, which is what the models actually read — the CSV copy is
for inspection and for anyone using the dataset outside this pipeline.

Eight families, per the plan: links (`n_urls`, `has_shortener`, `suspicious_tld`,
`ip_as_domain`, `url_digit_ratio`, `domain_len_max`, `brand_domain_mismatch`), then
`money`, `pressure`, `credentials`, `brands`, `optout`, `opener` — each as a `_n` count and
a `_hit` flag — then text shape (`length`, `n_words`, `capital_ratio`, `digit_ratio`,
`punct_ratio`, `n_exclaim`, `n_allcaps_words`, `has_leetspeak`, `avg_word_len`).

Two things the measured means show:

```
label       feat_length  feat_n_urls  feat_pressure_hit  feat_optout_hit
ham              59.278        0.010              0.005            0.000
spam            127.570        0.215              0.029            0.004
smishing        147.127        0.341              0.085            0.032
```

* `length` is the strongest single feature, and ham averages 59 characters against 147 —
  but that gap is partly an artefact of the corpora, since NUS is casual student chat. A
  real inbox has long legitimate messages this corpus lacks.
* `optout_hit` runs the opposite way to the plan's prediction: 3.2% of smishing carries
  "Reply STOP" against 0.4% of advertisement spam. Scammers mimic it.

17 of the 29 get exactly zero split gain in the trained model — every money / pressure /
credentials / brands / opt-out / opener flag. They are populated, the model just never
splits on them, because a Qwen embedding already encodes that semantics. Text shape and
link structure are what earn their place.

## Loading it

```python
import pandas as pd
df = pd.read_csv("combined_sms_dataset.csv", keep_default_na=False, dtype={"text": str})

train = df[df["split"] == "train"]
test  = df[df["split"] == "test"]           # clean labels only, 354 positives
y = (df["label"] != "ham").astype(int)      # same as df["scam"]
```

`keep_default_na=False` matters: without it, blanks become `NaN` and text like `"NA"` or
`"null"` is silently turned into a missing value.
