# Stage-2 hand-labelling

The stage-2 accuracy reported today (0.742 excluding `other`) is measured against labels
nobody annotated. `dataset/scam_types.yaml` says so up front: the plan assumed
hand-labelling, but SmishTank ships its own 10 categories, so the labels are a **mapping**.
Two bugs in that mapping are already known:

1. `category_map` resolves the type from SmishTank's category *before* the brand is
   consulted, and only `account alert` / `other` fall through to `brand_decides`. SmishTank
   filed 20 IRS-branded messages under `Finance/Crypto`, so they become
   `investment and crypto`. The model called 17 of them `government impersonation` — which
   is right, and the `irs` / `hmrc` entries in the yaml's own government brand list agree.
   Correcting just these moves overall accuracy 0.742 → 0.767 and
   `investment and crypto` 0.203 → 0.469.
2. `scam_types.py:59` falls through to `other` whenever a `brand_decides` row carries a
   brand that is not in the yaml lists — **98 rows**, 85 with a blank brand cell, plus
   Costco, Walmart, Geico, Oprah. They are stamped `other`, and `other` is then excluded
   from the headline. So ~14% of the scoreable pool is mislabelled *and* invisible to the
   metric.

These files replace the mapping with annotation for the rows where it matters most.

## Files

| File | Rows | What it is |
|---|---|---|
| `dataset_340.csv` | 340 | The full set. |
| `dataset_240.csv` | 240 | Minimum viable. A **strict subset** — same ids, so labels carry over if you finish the rest later. |
| `type_guide.md` | — | The rules. Keep it open; see below. |
| `answer_key.csv` | 340 | Mapped label + model prediction, joined by `id`. **Do not open until you are done.** |
| `build_labeling_sets.py` | — | Regenerates all of the above. Seeded 200, so it reproduces exactly. |
| `Sorted.numbers` | 340 | The categorisation pass, as received. Source artifact — read, not edited. |
| `categorized_261.csv` | 261 | Extracted from it, repaired. **Feeds the type head.** |
| `extract_categorized.py` | — | Does that extraction, and documents the three repairs. |
| `test_unlabelled.csv` | 188 | Blind. The **test-split** rows stage 2 sees. |
| `test_unlabelled_key.csv` | 188 | Their split, true scam flag and stage-1 score. **Do not open first.** |
| `build_test_set.py` | — | Regenerates both. |

## The test set, and why it is separate

Every stage-2 number in this project is measured on the type head's own split, and that
pool is **26% test rows already** — so there is no honest end-to-end figure for both
stages on held-out data, and there cannot be until the head stops training on test rows.

`test_unlabelled.csv` is the fix. Of 10,120 test rows only 199 carry a type; labelling
all 9,921 others is absurd, so it takes the population stage 2 actually meets:

| | rows | why |
|---|---|---|
| true scam, flagged | 144 | stage 2 sees these |
| ham, flagged | 33 | stage 2 sees these too, and should answer `not a scam` |
| true scam, missed | 11 | stage 1's misses, so recall keeps an honest denominator |

Those 33 also close a hole in `scam-classification/type_metrics.py`: ham has no
`scam_type_true`, so the `not a scam` class cannot currently be scored from the corpus
columns at all.

Label these, retrain excluding them, and stage 2 has a number worth publishing.

## The categorisation pass

`categorized_261.csv` is a first pass over the 261 rows that have text, and it is worth
being precise about what it is: per the workbook's own note, categories were assigned by
a **keyword/pattern classifier followed by a manual review of everything left in
`Other`**. That is not the blind annotation this file asks for below.

It is still a large improvement. It disagrees with `scam_type` on **136 of 261 rows**,
and the disagreements are exactly the two bugs listed above — the IRS rows filed under
`investment and crypto`, and all 98 that fell through the brand map. Feeding it to
`app-backend/distill.py` moves the type head from 0.805 to 0.830 accuracy and 0.752 to
0.769 macro-F1, scored on the same held-out rows.

What it does **not** do is settle anything. A head trained on classifier output partly
learns to imitate that classifier, and accuracy measured against those same labels
cannot see a mistake they share. The blind pass is still the thing that produces a
number you can publish, and the self-agreement check below is still the only route to
knowing what the ceiling is.

Nor does it add coverage. All 79 `NEEDS_SOURCING` rows are still empty — charity and
family emergency have zero messages, Medicare/health and utility shutoff have three
each. Those four types are deferred in `distill.py` rather than half-learnt.

Both sets are shuffled — labelling 180 delivery messages in a row turns into
pattern-matching on position instead of content.

## Composition

Counts are deliberately not uniform. 10 per class is enough to separate a broken class
(~0.50) from a working one (~0.85) but gives a ±25pp interval; 69 per class for ±10pp
would cost ~900 labels. 25 is the knee of the curve, so classes in dispute get 25–30 and
classes the model already handles get 13 to confirm rather than re-measure.

| Bucket | 340 | 240 | Why |
|---|---|---|---|
| fallback-`other` pool | 98 | 98 | mislabelled *and* unscored today — highest value per label |
| investment and crypto | 30 | 30 | mapping known-broken (IRS → crypto) |
| tech support | 30 | 30 | model known-broken (0.431; misses are Netflix ×24, Amazon ×20) |
| bank alert | 25 | — | shares the boundary tech support fails on |
| government impersonation | 25 | 2 | absorbs the crypto mislabels |
| delivery and toll | 13 | — | 1.000 at n=180 — confirm, don't re-measure |
| romance / prize and lottery / job offer | 13 each | — | already 0.76–0.88 |
| family emergency | 20 | 20 | **zero coverage in the corpus** |
| charity | 20 | 20 | **zero coverage in the corpus** |
| Medicare and health | 20 | 20 | **zero coverage in the corpus** |
| utility shutoff | 20 | 20 | **1 row in the corpus**, 19 to source |

The 2 `government impersonation` rows in the 240 set pad 238 → 240; nothing rides on them.

## The 79 empty rows

`family emergency`, `charity` and `Medicare and health` have **no rows at all** in
SmishTank and `utility shutoff` has one — `scam_types.yaml` lists them under
`uncovered` / `thin`. Right now stage 2 scores 9 of 13 types and reports it as if it were
the taxonomy. Closing that is the single biggest win here, bigger than the accuracy delta.

Those rows ship with an empty `text` and `flag = NEEDS_SOURCING`. They are **work orders,
not labelling tasks** — you have to go and find real examples first, and you obviously
can't blind-label a message you just went looking for. Source them, paste the text in, then
have them labelled in a **second pass** shuffled among the rest.

One caveat to carry into the write-up: hand-written scam examples are systematically
cleaner and more on-the-nose than real ones. Those four classes will read optimistic.
Report them flagged, not folded into the headline.

## How to label

**Blind.** You see only the text. Do not open `answer_key.csv` first. Checking whether an
existing label is "actually that thing" means deciding whether to *overturn* it, and people
overturn far less often than they'd independently disagree — you would have rubber-stamped
most of those 20 IRS rows.

**Against `type_guide.md`, not intuition.** The hard boundaries are definitional, not
factual, and `prompt.yaml` already chose. Applying a different standard scores the model
down for obeying its own spec.

**Calibrate first.** Label 20, then go back and check yourself for consistency. You will
find a rule you applied two different ways. Fix the wording in `type_guide.md` *and*
`prompt.yaml`, throw those 20 out, start clean. Skipping this means your first 50 labels use
a different standard than your last 50.

**Re-label 40 of them a week later, blind to the first pass.** One annotator gives no
agreement estimate, and this is the cheap fix. Your self-agreement rate is the ceiling on
the model's achievable score: if you only agree with yourself 85% of the time, a model at
85% is at human parity and further tuning is noise-chasing. That single number reframes
every result in the project.

Budget ~15–25s per message once calibrated: the 98-row fallback batch ≈ 35 min, the full
261 real rows ≈ 1.5–2 h.

## Scoring, once labels exist

Report **four** numbers, never one:

- **strict** (primary label only) vs **lenient** (`second_label` counts) — the gap is
  taxonomy ambiguity, not model error.
- **macro** vs **prevalence-weighted** — these differ by 11.6pp on current data (0.627 vs
  0.742), because the model is perfect on `delivery and toll` and that class is 26% of real
  traffic but 4% of this stratified set. Macro answers "which types are weak";
  prevalence-weighted answers "how will this perform in the field". A stratified set can
  only give you macro — keep the existing 695 mapped rows, bug-fixed, as the prevalence
  sample.
