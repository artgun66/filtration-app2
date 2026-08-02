"""Encode every message once and cache the vectors to disk.

Re-encoding 88.5k messages per experiment is the difference between a two-minute and
a ninety-minute iteration loop, so the cache is the point of this module. Vectors are
L2-normalised (the analysis plan asks for it, and it makes LightGBM's splits
scale-free across dimensions).

  python embed.py --model minilm            # ~minutes, the baseline arm
  python embed.py --model qwen --limit 1000 # timing probe before the real run
  python embed.py --model qwen              # the main encoder

fp16 is the default: measured on this GTX 1060 a 2048^3 matmul runs 4.83 ms in fp16
against 5.67 ms in fp32, so Pascal's crippled native-fp16 rate does not apply to the
cuBLAS path. It also halves VRAM. --fp32 is there if a model misbehaves.
"""
import argparse, os, time

import numpy as np
import pandas as pd

import modeling as M

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = M.DATASET_CSV
CACHE = os.path.join(HERE, "embeddings")

MODELS = M.CFG["encoders"]
BATCH_SIZE = M.CFG["embedding"]["batch_size"]
MAX_SEQ_LEN = M.CFG["embedding"]["max_seq_len"]


def cache_paths(name):
    return (os.path.join(CACHE, f"embeddings_{name}.npy"),
            os.path.join(CACHE, f"embeddings_{name}_ids.npy"))


def load_cached(name, ids):
    """Return the cached matrix aligned to `ids`, or None if the cache cannot serve it."""
    vec_p, id_p = cache_paths(name)
    if not (os.path.exists(vec_p) and os.path.exists(id_p)):
        return None
    cached_ids = np.load(id_p)
    vecs = np.load(vec_p)
    pos = {int(v): i for i, v in enumerate(cached_ids)}
    if not all(int(i) in pos for i in ids):
        return None
    return vecs[[pos[int(i)] for i in ids]]


def encode(name, texts, batch_size=BATCH_SIZE, fp32=False, max_seq_len=MAX_SEQ_LEN):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if (fp32 or device == "cpu") else torch.float16
    model = SentenceTransformer(MODELS[name], device=device,
                                model_kwargs={"torch_dtype": dtype})
    # Qwen3-Embedding advertises a 32k context. encode() sorts by length, so the
    # final batches hold the longest messages and a full-corpus run OOMs a 6 GB card
    # even though nothing here is long. SMS are short -- p99 is 320 characters and
    # the whole corpus tops out at 3,082 -- so 256 tokens truncates almost nothing.
    model.max_seq_length = min(model.max_seq_length, max_seq_len)
    t0 = time.time()
    # encode() sorts by length internally before batching, so short SMS are not
    # padded out to the length of the longest message in the corpus.
    # progress bar off: it emits a line per update, which floods a non-tty log
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    # Model load is excluded: it is a fixed ~30 s cost and extrapolating it over the
    # corpus is what made the first probe read 92 min instead of the real figure.
    return np.asarray(vecs, dtype=np.float32), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minilm", choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="encode only the first N rows (timing probe)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore an existing cache")
    a = ap.parse_args()

    df = pd.read_csv(SRC, keep_default_na=False, dtype={"text": str})
    full_n = len(df)
    if a.limit:
        # Sample, do not head(): the first rows are all SmishTank, the longest
        # messages in the corpus, so a head() probe over-estimates the full run.
        df = df.sample(n=min(a.limit, full_n), random_state=200)
    ids, texts = df["id"].to_numpy(), df["text"].tolist()

    if not a.force and not a.limit:
        hit = load_cached(a.model, ids)
        if hit is not None:
            print(f"cache hit: {hit.shape} for {a.model}, nothing to do")
            return

    vecs, dt = encode(a.model, texts, batch_size=a.batch_size, fp32=a.fp32,
                      max_seq_len=a.max_seq_len)
    rate = len(texts) / dt
    print(f"\nencoded {len(texts):,} messages in {dt/60:.1f} min "
          f"({rate:.0f} msg/s), dim {vecs.shape[1]}")

    if a.limit:
        print(f"probe only -- full corpus ({full_n:,}) would take ~{full_n/rate/60:.0f} min. "
              f"Cache not written; re-run without --limit.")
        return

    vec_p, id_p = cache_paths(a.model)
    os.makedirs(CACHE, exist_ok=True)
    np.save(vec_p, vecs)
    np.save(id_p, ids)
    print(f"saved {vec_p} ({vecs.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
