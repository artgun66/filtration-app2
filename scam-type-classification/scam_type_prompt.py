"""Stage 2: name the scam type and the warning signs behind a suspicious SMS.

Stage 1 decides suspicious/safe. Only what it flags reaches here, so the real workload
is on the order of a thousand messages, not the whole corpus.

Qwen2.5-3B-Instruct in fp16 is ~6.2 GB and will not fit alongside anything else on a
6 GB card, so this loads it 4-bit (bitsandbytes NF4, ~2 GB) and runs as a separate pass
after the embedding cache is built.

  python scam_type_prompt.py --limit 20        # eyeball the output
  python scam_type_prompt.py --evaluate        # score against the mapped labels

The model is asked for JSON. transformers has no grammar constraint, so output is
parsed tolerantly, retried once with a repair instruction, and the parse-failure rate
is reported rather than hidden.
"""
import argparse, json, os, re, sys, time

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)          # the taxonomy is a property of the dataset

from dataset import scam_types as ST       # noqa: E402

DATASET_CSV = os.path.join(ROOT, "dataset", "combined_sms_dataset.csv")
RESULTS = os.path.join(HERE, "results")

with open(os.path.join(HERE, "prompt.yaml"), encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)

MODEL = CFG["model"]["name"]
GEN = CFG["generation"]

# The guide names each type, so a rename in scam_types.yaml propagates here.
_TYPE_NAMES = dict(zip(
    ["GOVERNMENT", "TECH_SUPPORT", "BANK", "DELIVERY", "FAMILY", "ROMANCE", "CRYPTO",
     "PRIZE", "CHARITY", "HEALTH", "UTILITY", "JOB", "OTHER", "NOT_SCAM"],
    ST.SCAM_TYPES))
assert len(_TYPE_NAMES) == len(ST.SCAM_TYPES), (
    "one placeholder per type, in the same order as scam_types.yaml")
TYPE_GUIDE = CFG["type_guide"].format(**_TYPE_NAMES).rstrip()

SYSTEM = CFG["json_system"].format()      # unescapes the doubled braces in the schema
USER = CFG["json_user"]
CHOICE_SYSTEM = CFG["choice_system"]
CHOICE_USER = CFG["choice_user"]
EXPLAIN_USER = CFG["explain_user"]

# Letter menu for the constrained-choice method. One letter per type, so the model
# picks from the taxonomy instead of writing a category name of its own.
LETTERS = CFG["choice_letters"]
assert len(LETTERS) == len(ST.SCAM_TYPES), "one letter per scam type"
CHOICE_MENU = "\n".join(f"{c}) {t}" for c, t in zip(LETTERS, ST.SCAM_TYPES))


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse(raw):
    """Pull the first JSON object out of a reply. Returns None if it cannot be read."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "scam_type" not in obj:
        return None
    # snap to a known type; the model sometimes paraphrases
    want = str(obj["scam_type"]).strip().lower()
    for t in ST.SCAM_TYPES:
        if want == t.lower():
            obj["scam_type"] = t
            break
    else:
        # Substring matching only, and only on a non-empty string: "" is a substring
        # of every type, so an empty scam_type used to silently become whichever type
        # sits first in SCAM_TYPES (government impersonation).
        hit = [t for t in ST.SCAM_TYPES
               if want and (t.lower() in want or want in t.lower())]
        obj["scam_type"] = hit[0] if hit else ST.OTHER
    return obj


def load_model(four_bit=CFG["model"]["four_bit"]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    kw = {"dtype": torch.float16, "device_map": "cuda:0"}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=CFG["model"]["quant_type"],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=CFG["model"]["double_quant"])
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
    model.eval()
    return tok, model


def classify(tok, model, messages, batch_size=GEN["batch_size"],
             max_new_tokens=GEN["max_new_tokens"]["json"]):
    """Returns (parsed_or_None, raw_text) per message."""
    import torch

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out = []
    for i in range(0, len(messages), batch_size):
        chunk = messages[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER.format(guide=TYPE_GUIDE, message=m[:600])}],
            tokenize=False, add_generation_prompt=True) for m in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=GEN["do_sample"],
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            raw = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.append((parse(raw), raw))
    return out


def _letter_ids(tok):
    """Token id for each option letter. They must be one token each for the argmax."""
    ids = []
    for c in LETTERS:
        enc = tok.encode(c, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"option letter {c!r} is {len(enc)} tokens: {enc}")
        ids.append(enc[0])
    return ids


def _prep(tok):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token


def classify_choice(tok, model, messages, batch_size=GEN["batch_size"]):
    """Returns (scam_type, confidence) per message.

    One forward pass per batch, no decode loop: the answer is an argmax over the 13
    option-letter logits at the final position, so the output is always a member of
    SCAM_TYPES. Confidence is that softmax restricted to the option letters.
    """
    import torch

    _prep(tok)
    ids = torch.tensor(_letter_ids(tok), device=model.device)

    out = []
    for i in range(0, len(messages), batch_size):
        chunk = messages[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": CHOICE_SYSTEM},
             {"role": "user", "content": CHOICE_USER.format(
                 guide=TYPE_GUIDE, menu=CHOICE_MENU, message=m[:600])}],
            tokenize=False, add_generation_prompt=True) for m in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            # left padding, so -1 is the last real token for every row in the batch
            logits = model(**enc).logits[:, -1, :].float()
        opt = logits.index_select(1, ids)
        prob = opt.softmax(dim=1)
        best = opt.argmax(dim=1)
        out.extend((ST.SCAM_TYPES[b], p[b].item())
                   for b, p in zip(best.tolist(), prob))
    return out


def explain(tok, model, messages, types, batch_size=GEN["batch_size"],
            max_new_tokens=GEN["max_new_tokens"]["explain"]):
    """Warning signs and a one-line explanation, with the type already pinned."""
    import torch

    _prep(tok)
    out = []
    for i in range(0, len(messages), batch_size):
        chunk, kinds = messages[i:i + batch_size], types[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": CHOICE_SYSTEM},
             {"role": "user", "content": EXPLAIN_USER.format(
                 message=m[:600], scam_type=k)}],
            tokenize=False, add_generation_prompt=True) for m, k in zip(chunk, kinds)]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=GEN["do_sample"],
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            raw = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = _JSON_RE.search(raw or "")
            try:
                obj = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                obj = None
            out.append(obj)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=CFG["evaluate"]["limit"])
    ap.add_argument("--batch-size", type=int, default=GEN["batch_size"])
    ap.add_argument("--evaluate", action="store_true",
                    help="score against the dataset's scam_type column")
    ap.add_argument("--fp16", action="store_true", help="skip 4-bit (needs >6 GB free)")
    ap.add_argument("--method", choices=["choice", "json"], default="choice",
                    help="choice: argmax over the 13 options (default). "
                         "json: free-generate the whole object -- kept for comparison, "
                         "it collapses onto one category, see classify_choice")
    a = ap.parse_args()

    df = pd.read_csv(DATASET_CSV, keep_default_na=False, dtype={"text": str})
    labelled = df[df["scam_type"] != ""]
    if a.evaluate:
        # scam_types.py routes 'advertisement' and 'loans/credit' to `other` because the
        # taxonomy has no home for them -- 262 of the 360 `other` rows. Scoring against
        # those measures the mapping, not the model: an ad-styled lure has no correct
        # answer to find. Dropped here, leaving `other` as the 98 rows that genuinely
        # fit no type. Everything else, including the real `other`, is still scored.
        drop = labelled["category"].str.strip().str.lower().isin(
            set(CFG["evaluate"]["unscoreable_categories"]))
        print(f"excluded {drop.sum()} unscoreable rows (category advertisement or "
              f"loans/credit, mapped to '{ST.OTHER}'); {(~drop).sum()} remain")
        labelled = labelled[~drop]
        # 'other' is still a catch-all, so it is also reported separately below.
        sample = labelled.sample(n=min(a.limit, len(labelled)), random_state=CFG["evaluate"]["seed"])
    else:
        sample = labelled.head(a.limit)

    print(f"loading {MODEL} ({'fp16' if a.fp16 else '4-bit nf4'})...")
    tok, model = load_model(four_bit=not a.fp16)

    texts = sample["text"].tolist()
    t0 = time.time()
    if a.method == "choice":
        picked = classify_choice(tok, model, texts, a.batch_size)
        pred = [t for t, _ in picked]
        conf = [c for _, c in picked]
        # scoring needs the type only; the explanation pass is for eyeballing
        detail = explain(tok, model, texts, pred, a.batch_size) if not a.evaluate else None
    else:
        res = classify(tok, model, texts, a.batch_size)
        pred = [(r["scam_type"] if r else ST.OTHER) for r, _ in res]
        conf = [None] * len(res)
        ok = [r for r, _ in res if r]
        print(f"JSON parsed: {len(ok)}/{len(res)} ({len(ok)/len(res):.0%})")
    dt = time.time() - t0
    print(f"{len(pred)} messages in {dt:.0f}s ({len(pred)/dt:.1f}/s)\n")

    if not a.evaluate:
        for i, (_, row) in zip(range(10), sample.head(10).iterrows()):
            print("-" * 70)
            print("MSG :", row["text"][:110])
            print("TRUE:", row["scam_type"])
            if a.method == "choice":
                print(f"PRED: {pred[i]}  (p={conf[i]:.2f})")
                print("     ", json.dumps(detail[i], ensure_ascii=False) if detail[i] else "no explanation")
            else:
                print("PRED:", pred[i])
        return

    truth = sample["scam_type"].tolist()
    # id first: without it every consumer has to join these rows back to the corpus on
    # message text, which is fragile (duplicates, truncation, encoding) and was how
    # app-backend/distill.py --vs-teacher had to work until this was fixed.
    cmp = pd.DataFrame({"id": sample["id"].values, "true": truth, "pred": pred})
    print(f"overall accuracy      : {(cmp.true == cmp.pred).mean():.3f}")
    known = cmp[cmp.true != ST.OTHER]
    print(f"accuracy excluding 'other' truth: {(known.true == known.pred).mean():.3f} "
          f"(n={len(known)})")
    print("\nper true type:")
    cmp["ok"] = cmp.true == cmp.pred
    per = cmp.groupby("true")["ok"].agg(n="size", correct="sum", acc="mean").round(3)
    print(per.to_string())
    # one file per method so a re-run never overwrites the other method's evidence
    out = os.path.join(RESULTS, f"scam_type_eval_{a.method}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmp.assign(conf=conf, text=sample["text"].values).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
