# app-backend

The Python that builds what the phone ships. Everything here produces artifacts for
`app/`; nothing in the research tree depends on it.

```
predict.py       classify one message -- the reference the app must match
export_onnx.py   encoder + heads -> app/assets/models/, plus the test fixture
distill.py       trains the scam-type head that replaces the 3B LLM
```

## Why this is separate from the research code

`dataset/`, `scam-classification/`, `scam-type-classification/` and `labeling/` answer
"which model is best". This folder answers "what does the phone run", and the two have
different answers: the research best is `qwen_feat_clean` (0.6B encoder, ~1.2 GB), and
the phone runs `minilm_feat` (22M encoder, 86 MB) for 0.0003 less AUC.

Separate **files**, not separate code. These scripts still
`import modeling`, `run_arms` and `dataset.features` — copying the 29 feature
definitions, `ENCODER_FEAT` or `threshold_for` into this folder would create a second
source of truth for the exact values that must never drift between training and
serving. The dependency runs one way: app-backend → research, never back.

## Rebuilding the app's models

```bash
conda activate cyberscout

# 1. sanity: does single-message serving still reproduce the published metrics?
python predict.py --arm minilm_feat --verify

# 2. the scam-type head (needs scam-classification/results/model_minilm_feat.pkl)
python distill.py --train --arm minilm_feat

# 3. export everything the app bundles
python export_onnx.py --arm minilm_feat
python export_onnx.py --arm minilm_feat --check

# 4. the app's own tests, from ../app
cd ../app && npm test
```

Steps 1 and 3's `--check` are not ceremony. The 29 features exist in Python, ONNX and
TypeScript, and every way they can disagree is silent — a drifted regex or a
misaligned column changes the answer without raising anything.

## Try a message

```bash
python predict.py "URGENT: your account is locked, verify at http://bit.ly/x"
python predict.py --arm minilm_feat "Are we still on for lunch tomorrow?"
```

Without `--arm` you get `default_arm` from `scam-classification/training.yaml`, which
is still `qwen_feat_clean` — the research pick, deliberately left alone so
`annotate.py` and the published numbers do not move. Pass `--arm minilm_feat` to see
what the phone will say.

## Extending the scam-type head

`distill.py --train` uses 964 rows: the SmishTank rows that carry a usable type, 136
labels corrected from `../labeling/categorized_261.csv`, and 200 ham rows for
`not a scam`. Most of the correction gain is rows rescued out of the excluded `other`
bucket, and it is worth 0.805 → 0.830 accuracy (`--compare` scores both on the same
held-out rows).

Those corrections are classifier-assigned, not blind annotation, so the number measures
agreement with a better labeller rather than correctness — see `../labeling/README.md`.

`--vs-teacher` settles the question the head exists to answer. Scored on the same
held-out scam rows against the same corrected labels, the 3B teacher gets 0.702
accuracy / 0.618 macro-F1 and the head 0.817 / 0.775. The published 0.742 was measured
against the uncorrected mapping and is not comparable to either.

Eight of the plan's thirteen types have enough data to learn. Family emergency and
charity have no rows at all, Medicare/health and utility shutoff have three each; all
four are listed in `DEFERRED` and the head returns `null` instead of guessing.

## `not a scam`

A ninth class, and the only one whose training rows are ham. The head runs *only* on
messages stage 1 has already flagged, so before this it had no way to say "the filter
was wrong" — a false positive had to be forced into a scam category, and the app named
a kind of scam for a message that was not one.

It is trained on the **200 ham rows stage 1 scores highest**, not on random ham. That
matters: stage 1 reaches 1.000 accuracy on its own training split, so every train-split
ham row sits at p≈0.000 and is trivially separable. The hard rows — its actual false
positives — live in val and test. `--ham-cost` prices the trade:

| | scam-type accuracy | genuine scams called `not a scam` |
|---|---|---|
| without the class | 0.822 | 0 / 191 |
| with it | 0.817 | 6 / 191 |

For that it gets precision 0.88 and recall 0.88 on 50 held-out ham rows.

It **does not overturn stage 1**. The manifest exports the class name as
`type_not_scam`, the app never renders it as a scam type, and the verdict stands —
stage 1 is AUC 0.996 against this head's 0.83, and trading a loud false alarm for a
silent miss is the wrong direction for this audience.

`distill.py --pseudo-label --limit N` extends coverage by running the 3B teacher over
untyped scam rows (~2.9 s each, so 4,000 rows is about 3 hours). It is resumable —
ids already in `results/pseudo_labels.csv` are skipped. Then
`distill.py --train --with-pseudo`.

That does not fix the four empty types. Nothing in the corpus does; that is what
`../labeling/` is for.
