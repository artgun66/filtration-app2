"""Replace the 3B scam-type LLM with a head small enough to ship on a phone.

Stage 2 used Qwen2.5-3B because it wrote prose explanations. The app needs only the
type, and a 13-way label does not justify 2 GB of weights and 2.9 s per message. This
fits a linear head on the representation stage 1 already computes, so the app carries
one encoder and two heads.

  python distill.py --train                    # fit + evaluate, minutes
  python distill.py --pseudo-label --limit 4000  # extend coverage with the 3B, hours
  python distill.py --train --with-pseudo

What limits this is labels, not modelling. Of 29,653 scam rows only 1,055 carry a
type, and they come from SmishTank's own categories:

  delivery and toll 180, bank alert 116, tech support 102, government impersonation 85,
  romance 65, investment and crypto 64, prize and lottery 57, job offer 25,
  utility shutoff 1, other 360

Three of the plan's thirteen types -- family emergency, charity, Medicare and health --
have no rows at all, and utility shutoff has one. A classifier cannot learn a class it
has never seen, so the head covers eight types and returns None elsewhere rather than
guessing. labeling/ exists to fill exactly that hole.

`other` is excluded from training. Per labeling/README.md it is 262 genuine
advertisement/loan rows plus 98 that fell through the brand map -- 27% contaminated --
so it teaches the head to reproduce a known bug. Low-confidence predictions return
None instead, which is the same answer without the false precision.
"""
import argparse, os, pickle, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scam-classification"))

import modeling as M                            # noqa: E402
import run_arms as R                            # noqa: E402

RESULTS = os.path.join(HERE, "results")
PSEUDO = os.path.join(RESULTS, "pseudo_labels.csv")
EXCLUDE = {"other"}          # contaminated bucket, see the module docstring
MIN_PER_CLASS = 10           # below this a class cannot be evaluated, let alone learnt


def head_path(arm):
    return os.path.join(M.RESULTS, f"type_head_{arm}.pkl")


# ---------------------------------------------------------------- training data
def training_rows(df, with_pseudo=False):
    """(ids, labels, provenance) for every row that carries a usable type."""
    typed = df[(df["scam_type"] != "") & (~df["scam_type"].isin(EXCLUDE))]
    frame = pd.DataFrame({"id": typed["id"].to_numpy(),
                          "type": typed["scam_type"].to_numpy(),
                          "src": "mapped"})

    if with_pseudo:
        if not os.path.exists(PSEUDO):
            raise SystemExit(f"no {PSEUDO} -- run: python distill.py --pseudo-label")
        ps = pd.read_csv(PSEUDO)
        ps = ps[(~ps["type"].isin(EXCLUDE)) & (~ps["id"].isin(frame["id"]))]
        frame = pd.concat([frame, pd.DataFrame({"id": ps["id"], "type": ps["type"],
                                                "src": "pseudo"})], ignore_index=True)

    counts = frame["type"].value_counts()
    thin = counts[counts < MIN_PER_CLASS]
    if len(thin):
        print(f"dropping {len(thin)} class(es) under {MIN_PER_CLASS} rows: "
              + ", ".join(f"{t} ({n})" for t, n in thin.items()))
        frame = frame[frame["type"].isin(counts[counts >= MIN_PER_CLASS].index)]
    return frame


def train(arm, with_pseudo=False, seed=M.SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    df = M.load()
    frame = training_rows(df, with_pseudo)
    print(f"\n{len(frame)} labelled rows over {frame['type'].nunique()} types "
          f"({(frame['src'] == 'pseudo').sum()} pseudo)")
    print(frame["type"].value_counts().to_string())

    # Same representation as stage 1, from the same cache -- the app computes this
    # vector once and feeds both heads.
    X_all, names = R.feature_matrix(arm, df, df["id"].to_numpy())
    pos = {int(i): k for k, i in enumerate(df["id"].to_numpy())}
    X = X_all[[pos[int(i)] for i in frame["id"]]]
    y = frame["type"].to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed,
                                          stratify=y)
    # Linear, not boosted: ~700 rows against 400+ dimensions is where a tree ensemble
    # memorises. class_weight balances delivery-and-toll (180) against job offer (25).
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             random_state=seed)
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    print(f"\nheld-out accuracy: {(pred == yte).mean():.3f}  (n={len(yte)})")
    print("\n" + classification_report(yte, pred, zero_division=0))
    print("confusion matrix (rows = true):")
    labs = sorted(set(yte))
    cm = pd.DataFrame(confusion_matrix(yte, pred, labels=labs), index=labs, columns=labs)
    print(cm.to_string())

    with open(head_path(arm), "wb") as fh:
        pickle.dump({"model": clf, "classes": list(clf.classes_),
                     "feature_names": names, "arm": arm}, fh)
    print(f"\nwrote {head_path(arm)}")
    print("NOTE: scored against the mapped labels, which labeling/README.md documents "
          "as buggy.\n      Re-score on the hand labels once labeling/dataset_340.csv "
          "is filled in.")
    return clf


# ---------------------------------------------------------------- pseudo-labelling
def pseudo_label(limit, batch_size=None):
    """Run the 3B teacher over untyped scam rows and append to pseudo_labels.csv.

    Resumable: ids already in the file are skipped, so this can be run in chunks. At
    ~2.9 s/message the whole 28.6k untyped scam pool is roughly a day, which is why
    it is sampled rather than exhaustive.
    """
    import scam_type_prompt as S

    df = M.load()
    pool = df[(df["scam"] == 1) & (df["scam_type"] == "")]
    done = set()
    if os.path.exists(PSEUDO):
        done = set(pd.read_csv(PSEUDO)["id"].astype(int))
        pool = pool[~pool["id"].isin(done)]
    if pool.empty:
        print("nothing left to label")
        return
    # Sampled, not head(): the pool is ordered by source, so head() would label
    # Smishing_Dataset templates only.
    pool = pool.sample(n=min(limit, len(pool)), random_state=M.SEED)
    print(f"{len(done)} already labelled; labelling {len(pool)} more "
          f"(~{len(pool) * 2.9 / 3600:.1f} h)")

    tok, model = S.load_model()
    kw = {"batch_size": batch_size} if batch_size else {}
    out = S.classify_choice(tok, model, pool["text"].tolist(), **kw)

    new = pd.DataFrame({"id": pool["id"].to_numpy(),
                        "type": [t for t, _ in out],
                        "conf": [c for _, c in out]})
    os.makedirs(RESULTS, exist_ok=True)
    new.to_csv(PSEUDO, mode="a", header=not os.path.exists(PSEUDO), index=False)
    print(f"\nappended {len(new)} rows to {PSEUDO}")
    print(new["type"].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=M.CFG["default_arm"], choices=M.CFG["replayable"])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--pseudo-label", action="store_true")
    ap.add_argument("--with-pseudo", action="store_true",
                    help="include pseudo_labels.csv in the training set")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=0)
    a = ap.parse_args()

    if a.pseudo_label:
        pseudo_label(a.limit, a.batch_size or None)
    if a.train:
        train(a.arm, a.with_pseudo)
    if not (a.train or a.pseudo_label):
        ap.error("pass --train or --pseudo-label")


if __name__ == "__main__":
    main()
