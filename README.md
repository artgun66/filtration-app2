# Cyber Scout — suspicious-SMS detection

Two stages, per `Cyber-Scout-Analysis-Plan.docx`:

1. a lightweight classifier decides **safe vs suspicious**;
2. anything flagged goes to an LLM that names the **scam type** and the warning signs,
   in language an older adult can act on.

This repository covers stage 1 end to end, stage 2's prompt, and the on-device model
the phone app runs.

**The app runs both stages offline on the phone.** The 0.6B encoder behind the
research-best arm is too large to ship, so the app uses `minilm_feat` -- MiniLM plus
the 29 engineered features -- which costs 0.6 points of recall at 0.90 precision
against `qwen_feat_clean` while being roughly a twentieth of the size. Stage 2's 3B
LLM is replaced by a distilled linear head, because the app needs the scam *type* and
not the written explanation the LLM was there to produce.

| | research best | shipped on the phone |
|---|---|---|
| stage 1 | `qwen_feat_clean`, 0.6B encoder, ~600 MB | `minilm_feat`, 22M encoder, 90 MB |
| recall @ precision 0.90 | 0.944 | 0.938 |
| stage 2 | Qwen2.5-3B, 2.9 s/message | linear head, ~5K parameters |
| stage 2 accuracy | 0.702 | 0.838 |

Stage 2's row is not a typo: the 5K-parameter head beats the 3B LLM it replaces, scored
on the same held-out rows against the same corrected labels (`distill.py --vs-teacher`).
The teacher was never fine-tuned and answers over all 13 types, 12 of 191 times outside
the 8 the corpus can support — the head simply cannot make that class of mistake.

v1 checks a message the user shares or pastes, so it needs no SMS permissions at all.
Filtering messages as they arrive is a later step -- it needs Play policy review on
Android, and on iOS the filter extension only sees unknown senders and cannot show an
explanation.

## Layout

One folder per stage, plus the dataset they share. Everything tunable — word lists,
prompts, hyper-parameters — is in a `.yaml` next to the code that reads it.

```
dataset/                     the corpus and everything that defines it
  archives/                  the five sources exactly as downloaded
  raw/                       unpacked, each under its dataset name
  merge_datasets.py          unpack -> merge -> dedupe -> resolve labels -> dup_groups
  features.py                the 29 engineered features
  keywords.py                the arm-1 keyword rule
  scam_types.py              SmishTank categories -> the plan's 13 scam types
  sources.yaml               the 5 sources: links, archives, priorities
  features.yaml              feature word lists
  keywords.yaml              the 38 keyword terms
  scam_types.yaml            the taxonomy, category map and brand lists
  combined_sms_dataset.csv   88,542 rows x 94 columns
  combined_sms_dataset.md    what every column means

scam-classification/         stage 1: safe vs suspicious
  modeling.py                split, metrics, threshold selection, Bayesian search
  embed.py                   sentence embeddings with a disk cache
  run_arms.py                trains and scores all 7 arms
  annotate.py                fills the dataset's split and model_pred columns
  training.yaml              seed, arms, search spaces, encoder names
  cyber_scout_analysis.ipynb comparison table, SHAP, operating-point curve
  embeddings/  features.parquet  results/

scam-type-classification/    stage 2: which kind of scam
  scam_type_prompt.py        Qwen2.5-3B-Instruct, 4-bit
  prompt.yaml                model, generation settings and every prompt
  results/

example-code/                analysis_final.ipynb, the notebook this was ported from
```

The four folders above are the research. The four below are the product, kept apart so
neither is read as the other — the research best and what the phone runs are different
models, for size reasons, and mixing them invites confusion about which numbers apply.

```
app-backend/                 the Python that builds what the app ships
  predict.py                 classify one message; --verify replays the test split
  export_onnx.py             encoder + heads -> app/assets/models/, plus the fixture
  distill.py                 trains the type head that replaces the 3B LLM

core/                        the pipeline, once
  model.ts                   tokenize -> encode -> features -> heads
  features.ts  tokenizer.ts  the 29 features and WordPiece, ported from Python
  copy.ts                    every sentence the app says to the user

app/                         the Expo app (Android, iOS)
  App.tsx  src/  test/       paste box, share target, conformance tests
  assets/models/             what gets bundled into the APK

web/                         the browser app (any phone, no app store)
  index.html  src/  test/    one screen, Cache API, PWA
```

The dependency runs one way: `app-backend/` imports `modeling`, `run_arms` and
`dataset.features` rather than copying them, so the 29 feature definitions and the
fitted thresholds have exactly one source of truth. Nothing in the research tree
imports the app.

`core/` is the same idea one level down. `core/model.ts` takes the ONNX runtime as an
argument rather than importing one, so a single copy of the pipeline runs under
`onnxruntime-react-native` on a phone, `onnxruntime-web` in a browser and
`onnxruntime-node` in the tests. Three ports of the 29 features that could silently
disagree was already the sharpest risk in this project; a fourth was not worth having.

`dataset/` is the only importable package — stage 1 reads its features and keyword
rule, stage 2 reads its taxonomy. Nothing goes the other way.

## Running it

```bash
conda activate cyberscout

cd dataset
python merge_datasets.py          # ~1 min   -> combined_sms_dataset.csv (88,542 rows)
python features.py                # ~1 min   -> ../scam-classification/features.parquet

cd ../scam-classification
python embed.py --model minilm    # ~1 min
python embed.py --model qwen      # ~25 min  -> the main encoder
python modeling.py                # metric self-test, instant
python run_arms.py --smoke        # ~10 min, checks the pipeline runs
python run_arms.py                # hours: 50 Bayesian-search fits per arm
python annotate.py                # ~1 min   -> fills split + model_pred in the CSV

cd ../scam-type-classification
python scam_type_prompt.py --limit 20        # eyeball stage 2
python scam_type_prompt.py --evaluate --limit 300
```

Then open `scam-classification/cyber_scout_analysis.ipynb`.

## Building for the phone

```bash
cd app-backend
python predict.py "your account is locked, verify at http://bit.ly/x"
python predict.py --arm minilm_feat --verify   # replays the test split through this path
python distill.py --train --arm minilm_feat    # the phone's type head, ~1 min
python export_onnx.py --arm minilm_feat        # -> ../app/assets/models/
python export_onnx.py --arm minilm_feat --check

cd ../app
npm install
npm test                                       # both conformance tests
npx expo prebuild --platform android
npx expo run:android                           # needs a JDK and the Android SDK

cd ../web                                      # the browser build, no toolchain needed
npm install
npm test                                       # the same pipeline through WASM
npm run dev
```

`--verify`, `--check` and `npm test` are not optional ceremony. The 29 features exist
three times over — Python, ONNX, TypeScript — and every way they can disagree is
silent: a drifted regex or a misaligned column changes the answer without raising
anything. `--verify` proves the Python serving path reproduces the metrics in
`results/arm_metrics.csv`; `--check` proves the exported graph matches; `npm test`
proves the TypeScript matches on all 10,720 feature values and 160 token sequences,
then runs the whole app pipeline through the real ONNX graphs and checks every verdict
against Python. `web/`'s own `npm test` repeats that last step under WebAssembly, which
is a separate kernel implementation and can degrade quietly where the native one would
refuse to load.

That last one caught a real defect. Dynamic int8 quantisation shrinks the encoder
90 MB → 23 MB, but LightGBM splits on hard thresholds, so a small perturbation in one
embedding dimension moves a row across a split and the ensemble follows. Measured over
the fixture, int8 flipped *"Dear Customer never disclose your banking password
username and PIN to ..."* from 0.032 to 0.943 — a real bank security notice, reported
as a scam with 94% confidence. fp32 flips nothing, so
`app/assets/models/manifest.json` points at `encoder_fp32.onnx` and the 67 MB stays.

See `app/README.md` for the phone app, `web/README.md` for the browser build and
`app-backend/README.md` for rebuilding what they ship.

## Two apps, one pipeline

`app/` needs a JDK and the Android SDK to build, and on iOS every install route —
TestFlight, ad-hoc, the App Store — requires Apple's $99/yr Developer Program. `web/`
has no gatekeeper: a URL, HTTPS, and Add to Home Screen. It costs a 100 MB first load
and roughly 3× the inference time (38 ms vs 11 ms per message on the same laptop), and
it cannot be a share target on iOS.

Both run `core/` unmodified. The choice is distribution, not capability.

## Environment

`conda create -n cyberscout python=3.11`, then torch **cu121**, plus `transformers`,
`sentence-transformers`, `lightgbm`, `bayesian-optimization`, `shap`, `scikit-learn`,
`pandas`, `optuna`, `datasketch`, `umap-learn`, `pyarrow`, `accelerate`, `bitsandbytes`.

Measured on the target box (Ryzen 5 2600, 16 GB, GTX 1060 6 GB):

* fp16 is **15% faster** than fp32 here (4.83 ms vs 5.67 ms on a 2048³ matmul). Pascal's
  crippled native-fp16 rate does not apply to the cuBLAS path, so the encoder runs fp16.
* Qwen3-Embedding advertises a 32k context. `encode()` sorts by length, so the last
  batches OOM a 6 GB card. `embed.py` caps sequence length at 256 tokens — the corpus
  is p99 = 320 *characters*, so nothing real is truncated.
* LightGBM runs on CPU with `n_jobs=6` (physical cores).
* Qwen2.5-3B-Instruct does not fit in fp16 alongside anything else; stage 2 loads 4-bit.

## Dataset

88,542 unique messages after exact-text dedupe, from the five sources the analysis plan
names. Sources are carried in the CSV under the plan's names, with the page each was
downloaded from in `source_link`, and unpacked into `dataset/raw/` under the same names.

| `source` | `source_link` | rows kept |
|---|---|---|
| SmishTank | https://smishtank.com | 1,055 |
| Smishing_Dataset | https://github.com/shaghayegh-hp/Smishing_Dataset | 73,012 |
| NUS SMS Corpus | https://github.com/kite1988/nus-sms-corpus | 13,782 |
| Mendeley SMS Phishing | https://data.mendeley.com/datasets/f45bkkt8pr/1 | 621 |
| UCI SMS Spam Collection | https://archive.ics.uci.edu/dataset/228/sms+spam+collection | 72 |

Rows kept is *after* dedupe, and dedupe keeps the highest-priority source, so the column
says nothing about how much each dataset contributed. Mendeley ships 5,971 rows (its file
is published as `Dataset_5971.csv`) and UCI 5,574; almost all of both are also inside
Smishing_Dataset, which merged them. 37,996 rows appear in more than one source, and
`also_in` lists the others.

```
label  ham       58,889
label  spam       6,623     advertisement, per the plan's mapping
label  smishing  23,030
```

Three binary columns split that: `spam_label` (1 on the 6,623 spam), `smishing_label` (1 on
the 23,030 smishing), and `scam` = either, which is the stage-1 target the models train on.
Only the union is trustworthy — a source that publishes "spam" without saying whether it is
phishing lands in `spam_label` by default, so that column is a residual rather than a clean
advertisement class. See `dataset/combined_sms_dataset.md`.

Two columns drive the evaluation design:

* **`is_clean_label`** — 0 for rows whose only source is `Smishing_Dataset`, whose labels
  were assigned by keyword matching. 36,484 rows are keyword-labelled, 52,058 have a
  clean source. Keyword rows train; they never appear in validation or test.
* **`dup_group`** — near-duplicate group. Smishing is templated, so the same scam recurs
  with a fresh URL and survives exact-text dedupe. Splits are made on the group.

Two more are filled by `annotate.py` rather than by the merge, since one needs the
splitter and the other a trained model:

* **`split`** — `train` / `val` / `test` from `modeling.py`'s `make_splits`. Deterministic at
  `SEED = 200`, so it is the same split `run_arms.py` scored against.
* **`model_pred`** — hard 0/1 from `qwen_feat_clean` at its validation-fitted threshold
  (0.461). Filled for every row, so on `split == "train"` it is in-sample and far too
  flattering; score it on `split == "test"` only. On train it flags 15,265 of 28,945
  positives, which is not a failure — that arm never saw the 36,484 keyword-labelled
  rows and disagrees with many of them.

## Results

All seven arms on one clean test set (10,120 messages, 354 of them suspicious). Threshold
fitted on validation to hold 95% recall, then applied unchanged.

| arm | AUC | recall | 95% CI | specificity | precision | missed | false alarms |
|---|---|---|---|---|---|---|---|
| keyword | 0.851 | 1.000 | — | 0.000 | 0.035 | 0 | 9,766 |
| tfidf | 0.989 | 0.946 | .922–.968 | 0.982 | 0.661 | 19 | 172 |
| minilm | 0.990 | 0.955 | .931–.975 | 0.953 | 0.424 | 16 | 460 |
| qwen | 0.995 | 0.938 | .912–.963 | 0.985 | 0.692 | 22 | 148 |
| qwen_feat | 0.996 | 0.938 | .911–.964 | 0.996 | 0.900 | 22 | 37 |
| ann | 0.996 | 0.896 | .859–.928 | 0.995 | 0.873 | 37 | 46 |
| **qwen_feat_clean** | **0.996** | 0.929 | .900–.956 | **0.998** | **0.951** | 25 | **17** |

Operating points for the best model:

| recall target | threshold | false alarms per 1,000 safe messages |
|---|---|---|
| 90% | 0.891 | 1 |
| 95% | 0.446 | 7 |
| 99% | 0.004 | 124 |

## Findings so far

* **A keyword rule cannot do this job.** To reach 95% recall it has to flag *every*
  message (specificity 0.000, AUC 0.851). Since a keyword rule is what labelled 82% of
  the raw corpus, that is also the reason for the clean-evaluation design.
* **The keyword-labelled rows can be dropped entirely.** This was the open question
  behind the "train on it, evaluate clean" decision, and the ablation answers it:
  removing all 36,484 of them **improved** every test number — AUC +0.0005 (noise, the
  recall intervals overlap almost completely), but false alarms fell 37 → 17 and
  precision rose 0.90 → 0.95, at the cost of 3 more missed scams out of 354. Training
  also got 3.5× faster (22 min → 6 min). They were not buying accuracy. **Recommendation:
  make `qwen_feat_clean` the default and treat the keyword source as optional.**
* **The plan's keyword features are redundant; its text-shape features are not.** SHAP
  and LightGBM split gain agree: of the 29 engineered features, 17 get *exactly zero*
  gain — every money / pressure / credentials / brands / opt-out / opener flag among
  them. Not a bug: those columns are populated (brands fires on 4,349 rows, pressure on
  2,424), the model simply never splits on them, because a Qwen embedding already
  encodes that semantics better than a binary flag. Meanwhile only 31 of 1,024 embedding
  dims go unused. What earned its place is text shape and link structure — `length` alone
  carries 10× the gain of the next feature. The features as a block matter a great deal
  (adding them cut false alarms from 148 to 37), just not the keyword half.
* **Message length may be doing too much work.** It is the single strongest engineered
  feature, and ham averages 59 characters against 147 for smishing — but that gap is
  partly an artefact of what the corpora are: NUS is casual student chat, and the scam
  set is curated screenshots. A real inbox contains long *legitimate* messages (bank
  notifications, appointment reminders) that this corpus has almost none of, so some of
  the measured specificity may not survive contact with real traffic.
* **Opt-out language points the opposite way to the plan's prediction.** The plan expects
  `Reply STOP` to mark legitimate marketing that scammers skip. Measured, it fires on
  3.2% of smishing versus 0.4% of advertisement spam — scammers mimic it. Still a useful
  feature, just with the sign reversed.
* **Scam-type labels were mostly already there, but wrong more often than expected.** The
  plan assumed hand-labelling from scratch; SmishTank supplied 1,055 categorised messages
  covering 10 of 13 types. A correction pass over the 261 rows in `labeling/` disagreed
  with the derived label on **136 of them** — the IRS-under-crypto bug and the 98-row
  brand-map fallback — and fixing those moves the type head from 0.775 to 0.838 accuracy.
  Family emergency and charity still have **no** examples and Medicare/health and utility
  shutoff have three each; no relabelling reaches those, only sourcing new messages.
* **The 95% recall target does not survive validation → test.** Six of seven arms fitted
  to exactly 95% recall on validation landed *below* it on test (0.896–0.955). That is
  the expected behaviour of picking the tightest threshold that just clears a bar, given
  ~354 positives on each side: the sampling error falls below the bar about half the
  time. Measured fix: **tune validation to 97.0% to land on 95% on test**, costing
  specificity 0.996 → 0.991. `cyber_scout_analysis.ipynb` §4 plots the curve.

### Stage 2

Qwen2.5-3B-Instruct, 4-bit, parses valid JSON on 6/6 of a first spot-check and gets 6/6
categories right. Two prompt bugs had to be fixed to get there, both worth keeping in
mind if the prompt is edited:

* **Tell the model the message is already flagged.** Without it, the model re-litigates
  whether the text is a scam at all and files every romance opener ("long time no see,
  I'm Aleen") under *other* — correctly, on the face of it, since the first message of a
  romance scam contains no link, no threat and no ask. Stating that an earlier filter
  already flagged the message changes the question from "is this bad?" to "which kind?"
  and fixed all three romance misses at once.
* **Say where adjacent categories divide.** An unpaid UPS customs fee was landing under
  *utility shutoff* on the strength of "unpaid fee". One sentence pinning courier fees to
  *delivery and toll* and household utilities to *utility shutoff* fixed it.

Throughput is ~17 s/message. bitsandbytes NF4 has no int4 acceleration on Pascal — it
dequantizes to fp16 per layer — so a Q4_K_M GGUF through llama.cpp/Ollama would be
substantially faster if stage 2 ever needs to run over more than a few hundred messages.

## Limits

* **Test rests on ~354 suspicious messages.** The clean sources are overwhelmingly NUS
  ham, so the clean pool holds only ~1,750 suspicious messages. Recall's 95% CI on this
  test set is about ±2 points. Precision and accuracy move with prevalence and are not
  comparable across splits; recall, specificity and AUC are.
* **The data is 2011–2023.** Nothing here has seen a scam newer than SmishTank's last
  submission, 2023-12-13. The 300+ collected messages named in the plan — the only held-out,
  current, local test set — are not in the project. Treat every number as an upper bound.
* **`sender_type` is deliberately not a model feature.** Only 955 of 88,542 rows have
  one and they all come from SmishTank, so a model given it would learn provenance
  rather than smishing. It belongs in the app as a rule.
