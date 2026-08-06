"""Export the chosen arm to the artifacts the phone app loads.

  python export_onnx.py --arm minilm_feat

Writes to serving/app_assets/:
  encoder_int8.onnx    tokens -> a 384-dim L2-normalised vector
  scam_head.onnx       that vector + the 29 features -> P(scam)
  type_head.onnx       the same input -> one of the covered scam types
  vocab.txt            WordPiece vocabulary for the tokenizer
  manifest.json        threshold, feature order, class names, dims
  golden_vectors.json  the conformance fixture -- see below

Pooling and L2 normalisation are inside the encoder graph on purpose. They are two
lines of Python and two easy-to-get-wrong lines of TypeScript; putting them in the
graph means the app cannot get them wrong at all.

golden_vectors.json is the point of this file. The 29 features have to be rewritten in
TypeScript, and Python `re` and JavaScript `RegExp` disagree on `\\b`, `\\w` and
Unicode case-folding -- all three of which dataset/features.py uses. A divergence there
throws no error; it quietly moves every probability. The fixture turns that silent
failure into a failing test.
"""
import argparse, json, os, pickle, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scam-classification"))

import modeling as M                            # noqa: E402
import run_arms as R                            # noqa: E402
from dataset import features as F               # noqa: E402
from dataset import keywords as K               # noqa: E402
import predict as P                             # noqa: E402

ASSETS = os.path.join(HERE, "app_assets")
N_GOLDEN = 200

# fp32, not int8, and the 67 MB is worth it. Measured with --check over the fixture:
#
#   encoder_fp32.onnx   max|delta| 0.0033   0 decision flips
#   encoder_int8.onnx   max|delta| 0.9108   1 decision flip
#
# LightGBM splits on hard thresholds, so a small perturbation in one embedding
# dimension does not stay small -- it moves the row to the other side of a split and
# the ensemble follows. The single int8 flip was
# "Dear Customer never disclose your banking password username and PIN to ..." going
# from 0.032 to 0.943: a real bank security notice, reported to the user as a scam
# with 94% confidence. That is the exact failure this app cannot afford, and 90 MB is
# unremarkable for a sideloaded prototype.
SHIPPED_ENCODER = "encoder_fp32.onnx"


def build_encoder_module(st_model):
    """Wrap transformer + mean-pool + L2-norm as one exportable module."""
    import torch
    from torch import nn

    class Enc(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            tok = out[0]                                   # last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(tok.dtype)
            # mean over real tokens only; clamp guards an all-padding row
            pooled = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, p=2.0, dim=1)

    return Enc(st_model[0].auto_model).eval()


def export_encoder(name, out_path):
    import torch
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from sentence_transformers import SentenceTransformer

    # CPU and fp32: the export traces the graph, and a CUDA/fp16 trace both mismatches
    # the CPU dummy inputs and bakes half-precision into a graph that will run int8.
    st = SentenceTransformer(M.CFG["encoders"][name], device="cpu")
    st.max_seq_length = min(st.max_seq_length, M.CFG["embedding"]["max_seq_len"])
    enc = build_encoder_module(st).float()

    dummy = torch.ones(1, 16, dtype=torch.long)
    fp32 = out_path.replace("_int8", "_fp32")
    torch.onnx.export(
        enc, (dummy, torch.ones_like(dummy)), fp32,
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "embedding": {0: "batch"}},
        opset_version=17)
    # Both are written, but the manifest points at fp32 -- see SHIPPED_ENCODER.
    quantize_dynamic(fp32, out_path, weight_type=QuantType.QInt8)

    tok = st.tokenizer
    tok.save_pretrained(ASSETS)
    # transformers saves only tokenizer.json for a fast tokenizer, but the app's
    # WordPiece implementation wants the plain vocabulary, index order preserved.
    vocab = tok.get_vocab()
    with open(os.path.join(ASSETS, "vocab.txt"), "w", encoding="utf-8") as fh:
        for token, _ in sorted(vocab.items(), key=lambda kv: kv[1]):
            fh.write(token + "\n")
    return fp32, st


def export_sklearn(model, n_features, out_path, zipmap=False):
    """LightGBM or logistic regression -> ONNX, float input of width n_features."""
    if type(model).__name__.startswith("LGBM"):
        # onnxmltools rejects skl2onnx's FloatTensorType even though the two are
        # structurally identical -- it checks the class, so the import must match.
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        # onnxmltools's LightGBM converter tops out at opset 15; the tree ops it emits
        # have not changed since, so this costs nothing.
        onx = convert_lightgbm(model, initial_types=[("input", FloatTensorType([None, n_features]))],
                               zipmap=False, target_opset=15)
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onx = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, n_features]))],
                              target_opset=17,
                              options={id(model): {"zipmap": False}})
    with open(out_path, "wb") as fh:
        fh.write(onx.SerializeToString())


def golden(clf, tokenizer=None, n=N_GOLDEN):
    """Feature values, token ids and final probability for a spread of real messages.

    Stratified over label and source so the fixture exercises URLs, brands, leetspeak
    and plain ham rather than 200 near-identical smishing templates. Token ids are
    included so the hand-written WordPiece tokenizer in the app is covered by the
    same test as the features -- it is the other half of the input the model sees.
    """
    df = M.load()
    take = (df.groupby(["label", "source"], group_keys=False)
              .apply(lambda g: g.sample(min(len(g), max(2, n // 12)),
                                        random_state=M.SEED))
              .head(n))
    texts = take["text"].tolist()
    probs = clf.probs(texts)
    ids = None
    if tokenizer is not None:
        ids = tokenizer(texts, truncation=True,
                        max_length=M.CFG["embedding"]["max_seq_len"])["input_ids"]
    rows = []
    for i, (t, p) in enumerate(zip(texts, probs)):
        f = F.extract_features(t)
        row = {"text": t,
               "features": {k: (round(v, 6) if isinstance(v, float) else v)
                            for k, v in f.items()},
               "keywords": K.flags(t),
               "prob": round(float(p), 6)}
        if ids is not None:
            row["input_ids"] = ids[i]
        rows.append(row)
    return rows


def check_onnx(threshold):
    """Run the exported int8 graph over the fixture and compare to Python.

    int8 quantisation is lossy by construction, so this is not expected to match
    exactly -- the question is whether it moves any *decision*. That is the number
    that matters, and it has to be measured rather than assumed.
    """
    import onnxruntime as ort
    from transformers import AutoTokenizer

    with open(os.path.join(ASSETS, "golden_vectors.json"), encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    with open(os.path.join(ASSETS, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)

    tok = AutoTokenizer.from_pretrained(ASSETS)
    enc_file = man.get("encoder_file", SHIPPED_ENCODER)
    enc = ort.InferenceSession(os.path.join(ASSETS, enc_file))
    head = ort.InferenceSession(os.path.join(ASSETS, "scam_head.onnx"))
    names = man["feature_names"]

    got, ref = [], []
    for c in cases:
        t = tok(c["text"], truncation=True, max_length=man["max_seq_len"],
                return_tensors="np")
        v = enc.run(None, {"input_ids": t["input_ids"].astype(np.int64),
                           "attention_mask": t["attention_mask"].astype(np.int64)})[0][0]
        vals = {**c["features"], **c["keywords"]}
        row = np.array([[v[int(n[3:])] if n.startswith("emb") else vals[n]
                         for n in names]], dtype=np.float32)
        out = head.run(None, {"input": row})
        probs = out[1]
        got.append(float(probs[0][1] if np.ndim(probs) > 1 else probs[0]))
        ref.append(c["prob"])

    got, ref = np.array(got), np.array(ref)
    d = np.abs(got - ref)
    flips = int(((got >= threshold) != (ref >= threshold)).sum())
    print(f"\n{enc_file} vs Python on {len(cases)} cases:")
    print(f"  max|delta| {d.max():.4f}   mean {d.mean():.5f}   decision flips {flips}")
    print("  OK" if flips == 0 else f"  {flips} case(s) cross the threshold after quantisation")
    return flips == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="minilm_feat", choices=M.CFG["replayable"])
    ap.add_argument("--golden-only", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="score the exported int8 graph against the fixture")
    a = ap.parse_args()

    if a.check:
        clf = P.Classifier(a.arm)
        raise SystemExit(0 if check_onnx(clf.threshold) else 1)

    os.makedirs(ASSETS, exist_ok=True)
    clf = P.Classifier(a.arm)
    print(f"arm {a.arm}: {len(clf.feature_names)} features, threshold {clf.threshold:.6f}")

    tok = None
    if a.golden_only and clf.encoder_name:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(M.CFG["encoders"][clf.encoder_name])

    if not a.golden_only:
        enc_name = clf.encoder_name
        if enc_name:
            fp32, st = export_encoder(enc_name, os.path.join(ASSETS, "encoder_int8.onnx"))
            tok = st.tokenizer
            print(f"  encoder {enc_name}: "
                  f"{os.path.getsize(fp32)/1e6:.0f} MB fp32 -> "
                  f"{os.path.getsize(os.path.join(ASSETS,'encoder_int8.onnx'))/1e6:.0f} MB int8")

        export_sklearn(clf.model, len(clf.feature_names),
                       os.path.join(ASSETS, "scam_head.onnx"))
        print(f"  scam_head.onnx  {os.path.getsize(os.path.join(ASSETS,'scam_head.onnx'))/1e6:.2f} MB")

        classes = None
        if clf.type_head:
            export_sklearn(clf.type_head["model"], len(clf.feature_names),
                           os.path.join(ASSETS, "type_head.onnx"))
            classes = clf.type_head["classes"]
            print(f"  type_head.onnx  {os.path.getsize(os.path.join(ASSETS,'type_head.onnx'))/1e6:.2f} MB")

        manifest = {
            "arm": a.arm, "encoder": clf.encoder_name,
            "encoder_file": SHIPPED_ENCODER,
            "threshold": clf.threshold,
            "max_seq_len": M.CFG["embedding"]["max_seq_len"],
            "feature_names": clf.feature_names,
            "engineered": F.FEATURE_NAMES, "keywords": K.COLUMNS,
            "feature_labels": F.FEATURE_LABELS,
            "type_classes": classes,
            "type_min_conf": 0.35,
            # The app must say so rather than silently mislabel: these have no
            # training rows at all. See labeling/README.md.
            "types_not_covered": ["family emergency", "charity",
                                  "Medicare and health", "utility shutoff"],
        }
        with open(os.path.join(ASSETS, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print("  manifest.json")

        # The word lists ship as data, not as retyped TypeScript. features.yaml stays
        # the single source of truth: edit it and re-export, and the app follows.
        with open(os.path.join(ASSETS, "feature_config.json"), "w", encoding="utf-8") as fh:
            json.dump({"known_tlds": sorted(F.KNOWN_TLDS),
                       "suspicious_tlds": sorted(F.SUSPICIOUS_TLDS),
                       "shorteners": sorted(F.SHORTENERS),
                       "mismatch_brands": list(F.MISMATCH_BRANDS),
                       "families": F.FAMILIES,
                       "keywords": list(K.KEYWORDS)}, fh, indent=1)
        print("  feature_config.json")

    rows = golden(clf, tok)
    with open(os.path.join(ASSETS, "golden_vectors.json"), "w", encoding="utf-8") as fh:
        json.dump({"arm": a.arm, "threshold": clf.threshold, "cases": rows}, fh, indent=1)
    print(f"  golden_vectors.json  {len(rows)} cases")
    print(f"\nwrote {ASSETS}")


if __name__ == "__main__":
    main()
