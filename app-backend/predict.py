"""Classify one message. The reference implementation the phone app must match.

Everything upstream of this is batch: run_arms.py trains over the whole corpus,
annotate.py replays over the CSV. Neither takes a string. This does:

  python predict.py "URGENT: your account is locked, verify at http://bit.ly/x"
  python predict.py --arm bge_feat --verify       # reproduce probs_<arm>_test.npy

The feature vector is assembled **by name** from the pickle's `feature_names`, never
by positional concatenation. A column-order mismatch between training and serving does
not raise -- it silently moves every probability -- so the order comes from the fitted
model itself and `--verify` proves the whole path reproduces the published numbers.
"""
import argparse, json, os, pickle, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scam-classification"))

import modeling as M                                    # noqa: E402
import run_arms as R                                    # noqa: E402
from annotate import threshold_for                      # noqa: E402
from dataset import features as F                       # noqa: E402
from dataset import keywords as K                       # noqa: E402

_ENCODERS = {}


def encoder(name):
    """The sentence encoder, loaded once per process.

    Settings mirror embed.py exactly -- same device, same dtype, same max_seq_len,
    same L2 normalisation. The dtype is not cosmetic: embed.py encodes in fp16 on
    CUDA, and serving in fp32 reproduces the cached vectors only to ~6e-4, which
    moves the odd borderline message across the threshold. --verify catches it.
    """
    if name not in _ENCODERS:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16
        m = SentenceTransformer(M.CFG["encoders"][name], device=device,
                                model_kwargs={"torch_dtype": dtype})
        m.max_seq_length = min(m.max_seq_length, M.CFG["embedding"]["max_seq_len"])
        _ENCODERS[name] = m
    return _ENCODERS[name]


class Classifier:
    """A trained arm, ready to score raw text."""

    def __init__(self, arm):
        path = os.path.join(M.RESULTS, f"model_{arm}.pkl")
        if not os.path.exists(path):
            raise SystemExit(f"no {path} -- run: python run_arms.py --arms {arm}")
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        self.arm = arm
        self.model = blob["model"]
        self.feature_names = blob["feature_names"]
        self.threshold = threshold_for(arm)
        # Optional: the distilled scam-type head, if one has been fitted for this arm.
        # It shares this arm's feature vector, so the app encodes once and runs both.
        self.type_head = None
        head_p = os.path.join(M.RESULTS, f"type_head_{arm}.pkl")
        if os.path.exists(head_p):
            with open(head_p, "rb") as fh:
                self.type_head = pickle.load(fh)
        # encoder+features arms map through ENCODER_FEAT; the embeddings-only arms
        # (minilm, qwen) are named after their encoder; feat_only has none.
        self.encoder_name = R.ENCODER_FEAT.get(arm) or (
            arm if arm in M.CFG["encoders"] else None)

    def matrix(self, texts):
        """(n, len(feature_names)) assembled by name, in the model's own order."""
        cols = {}
        if self.encoder_name:
            v = encoder(self.encoder_name).encode(
                texts, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False, batch_size=M.CFG["embedding"]["batch_size"])
            for i in range(v.shape[1]):
                cols[f"emb{i}"] = v[:, i]
        feats = [F.extract_features(t) for t in texts]
        for name in F.FEATURE_NAMES:
            cols[name] = np.array([f[name] for f in feats], dtype=np.float32)
        if any(n.startswith("kw_") for n in self.feature_names):
            flags = [K.flags(t) for t in texts]
            for name in K.COLUMNS:
                cols[name] = np.array([f[name] for f in flags], dtype=np.float32)

        missing = [n for n in self.feature_names if n not in cols]
        if missing:
            raise RuntimeError(f"{self.arm}: cannot build {len(missing)} features, "
                               f"e.g. {missing[:3]}")
        return np.column_stack([cols[n] for n in self.feature_names]).astype(np.float32)

    def probs(self, texts):
        return self.model.predict_proba(self.matrix(texts))[:, 1]

    def scam_type(self, X, min_conf=0.35):
        """(type, confidence) from the distilled head, or (None, None).

        None is a real answer here. The head covers eight of the plan's thirteen
        types -- family emergency, charity, Medicare and health and utility shutoff
        have no training rows -- so a message of one of those kinds can only be
        forced into a neighbouring class. Below min_conf, say nothing.
        """
        if self.type_head is None:
            return None, None
        p = self.type_head["model"].predict_proba(X)[0]
        k = int(np.argmax(p))
        return ((self.type_head["classes"][k], round(float(p[k]), 4))
                if p[k] >= min_conf else (None, round(float(p[k]), 4)))


def classify(text, arm=None, _cache={}):
    """One message in, one verdict out."""
    arm = arm or M.CFG["default_arm"]
    if arm not in _cache:
        _cache[arm] = Classifier(arm)
    c = _cache[arm]
    X = c.matrix([text])                      # encode once, feed both heads
    p = float(c.model.predict_proba(X)[0, 1])
    scam = p >= c.threshold
    # Only name a type for something already judged suspicious -- the head was never
    # shown a ham message, so its answer on one is meaningless.
    kind, kind_p = c.scam_type(X) if scam else (None, None)
    return {"scam": scam, "prob": round(p, 6),
            "threshold": round(c.threshold, 6), "arm": arm,
            "signals": signals(text),
            "type": kind, "type_prob": kind_p}


def signals(text):
    """The engineered features that fired, in the plan's plain-English wording.

    dataset/features.py already carries UI-ready labels ("Brand named but link does
    not match"), so the app does not need its own copy.
    """
    f = F.extract_features(text)
    return [F.FEATURE_LABELS[k] for k in F.FEATURE_NAMES
            if k in F.FEATURE_LABELS and isinstance(f[k], int) and f[k] > 0
            and (k.endswith("_hit") or k.startswith("has_") or k in
                 ("suspicious_tld", "ip_as_domain", "brand_domain_mismatch"))]


def verify(arm):
    """Re-score the test split through this path and compare to the saved probs.

    Proves single-message serving and the batch run that produced arm_metrics.csv are
    the same computation.

    The bar is agreement in *decisions and metrics*, not bit-exact floats. embed.py
    encodes in fp16 and encode() sorts by length, so a message sits in a different
    batch -- with different padding -- here than it did during training, and fp16
    rounding differs by up to ~3e-2 on a few rows. Demanding equal floats would fail
    forever on numerics that cannot move a reported number. Metrics rounded to three
    decimals is the honest check: it fails loudly on a real defect (a misaligned
    feature column shifts recall immediately) and tolerates what it should.
    """
    ref_p = os.path.join(M.RESULTS, f"probs_{arm}_test.npy")
    if not os.path.exists(ref_p):
        raise SystemExit(f"no {ref_p} -- run: python run_arms.py --arms {arm}")
    ref = np.load(ref_p)

    df = M.load()
    te = (M.make_splits(df) == "test").values
    c = Classifier(arm)
    got = c.probs(df.loc[te, "text"].tolist())
    y = df.loc[te, "y"].to_numpy()

    delta = np.abs(got - ref)
    flips = int(((ref >= c.threshold) != (got >= c.threshold)).sum())
    a = M.binary_metrics(y, ref, c.threshold)
    b = M.binary_metrics(y, got, c.threshold)
    keys = ("recall", "precision", "specificity")
    ok = all(round(a[k], 3) == round(b[k], 3) for k in keys)

    print(f"{arm}: n={len(ref)}  max|delta|={delta.max():.2e}  mean={delta.mean():.2e}  "
          f"decision flips={flips}")
    for name, m in (("recorded", a), ("serving ", b)):
        print(f"  {name}  " + "  ".join(f"{k} {m[k]:.4f}" for k in keys)
              + f"  TP {m['tp']} FP {m['fp']} FN {m['fn']}")
    print("MATCH" if ok else "MISMATCH -- serving disagrees with the trained pipeline")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="message to classify")
    ap.add_argument("--arm", default=M.CFG["default_arm"], choices=M.CFG["replayable"])
    ap.add_argument("--verify", action="store_true",
                    help="reproduce probs_<arm>_test.npy through this path")
    a = ap.parse_args()

    if a.verify:
        raise SystemExit(0 if verify(a.arm) else 1)
    if not a.text:
        ap.error("give a message, or --verify")
    print(json.dumps(classify(" ".join(a.text), a.arm), indent=2))


if __name__ == "__main__":
    main()
