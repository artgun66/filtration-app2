"""Train and score every model arm in the analysis plan on one common test set.

Slow (Bayesian search is ~50 LightGBM fits per arm), so results are cached to
results/ and the notebook reads them rather than recomputing.

  python run_arms.py --smoke                 # 4 BO iterations, validates the pipeline
  python run_arms.py --arms keyword tfidf minilm
  python run_arms.py                         # everything, hours

Arms, per the plan:
  1 keyword   rule-based floor -- this is the kind of rule that labelled Smishing_Dataset
  2 tfidf     TF-IDF + LightGBM
  3 minilm    all-MiniLM-L6-v2 embeddings + LightGBM
  4 qwen      Qwen3-Embedding-0.6B + LightGBM
  5 qwen_feat Qwen + the 29 engineered features + LightGBM
  6 ann       MLP on Qwen embeddings, tuned with Optuna
  7 qwen_feat_clean  arm 5 retrained without the keyword-labelled source (ablation)

Every arm shares one split, one target and one threshold rule, so only the
representation varies. The threshold is fitted on validation to hold 95% recall and
applied to test unchanged; test is never used to choose anything.
"""
import argparse, os, sys, time

import numpy as np
import pandas as pd

import modeling as M

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)          # the dataset package defines the features and the rule

from dataset import features as F           # noqa: E402
from dataset.keywords import KEYWORDS       # noqa: E402

RESULTS = M.RESULTS
ARMS = M.CFG["arms"]
PARQUET = os.path.join(HERE, "features.parquet")


def keyword_score(texts):
    """Fraction of the keyword list present, used as a pseudo-probability."""
    out = np.zeros(len(texts), dtype=np.float32)
    for i, t in enumerate(texts):
        low = (t or "").lower()
        out[i] = sum(k in low for k in KEYWORDS) / len(KEYWORDS)
    return out


def load_embeddings(name, ids):
    import embed
    hit = embed.load_cached(name, ids)
    if hit is None:
        raise SystemExit(f"no cached {name} embeddings -- run: python embed.py --model {name}")
    return hit


def feature_matrix(arm, df, ids):
    """Return (X, feature_names). X is dense unless the arm is TF-IDF."""
    if arm == "tfidf":
        return None, None                       # built per-split inside run_arm
    if arm == "minilm":
        v = load_embeddings("minilm", ids)
        return v, [f"emb{i}" for i in range(v.shape[1])]
    if arm in ("qwen", "ann"):
        v = load_embeddings("qwen", ids)
        return v, [f"emb{i}" for i in range(v.shape[1])]
    if arm in ("qwen_feat", "qwen_feat_clean"):
        v = load_embeddings("qwen", ids)
        feats = pd.read_parquet(PARQUET)
        feats = feats.set_index("id").loc[ids][F.FEATURE_NAMES].to_numpy(np.float32)
        return np.hstack([v, feats]), ([f"emb{i}" for i in range(v.shape[1])] + F.FEATURE_NAMES)
    raise ValueError(arm)


def run_arm(arm, df, split, init_points, n_iter, seed=M.SEED):
    ids = df["id"].to_numpy()
    y = df["y"].to_numpy()
    tr, va, te = (split == "train").values, (split == "val").values, (split == "test").values

    # Arm 7 drops the keyword-labelled rows from training only. val/test are already
    # clean, so this isolates the effect of training on them.
    if arm == "qwen_feat_clean":
        tr = tr & (df["is_clean_label"] == 1).values

    t0 = time.time()

    if arm == "keyword":
        p = keyword_score(df["text"].tolist())
        clf, names = None, None
        probs = {"train": p[tr], "val": p[va], "test": p[te]}

    elif arm == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        # fitted on train only: fitting on everything would leak test vocabulary
        tf = dict(M.CFG["tfidf"])
        tf["ngram_range"] = tuple(tf["ngram_range"])
        vec = TfidfVectorizer(**tf)
        Xtr = vec.fit_transform(df.loc[tr, "text"])
        Xva, Xte = vec.transform(df.loc[va, "text"]), vec.transform(df.loc[te, "text"])
        best, _ = M.tune_lgbm(Xtr, y[tr], Xva, y[va], init_points, n_iter, seed)
        clf = M.fit_lgbm(Xtr, y[tr], best, seed)
        names = vec.get_feature_names_out().tolist()
        probs = {k: clf.predict_proba(X)[:, 1]
                 for k, X in (("train", Xtr), ("val", Xva), ("test", Xte))}

    elif arm == "ann":
        X, names = feature_matrix(arm, df, ids)
        clf = None
        probs = train_ann(X[tr], y[tr], X[va], y[va], X[te], n_trials=max(4, n_iter // 2), seed=seed)

    else:
        X, names = feature_matrix(arm, df, ids)
        best, _ = M.tune_lgbm(X[tr], y[tr], X[va], y[va], init_points, n_iter, seed)
        clf = M.fit_lgbm(X[tr], y[tr], best, seed)
        probs = {k: clf.predict_proba(X[m])[:, 1]
                 for k, m in (("train", tr), ("val", va), ("test", te))}

    # threshold chosen on val, then applied unchanged to test
    thr = M.pick_threshold(y[va], probs["val"], M.MIN_RECALL)
    rows = []
    for name, mask in (("train", tr), ("val", va), ("test", te)):
        m = M.binary_metrics(y[mask], probs[name], thr)
        m["arm"], m["split"], m["n"] = arm, name, int(mask.sum())
        if name == "test":
            lo, hi = M.bootstrap_ci(y[mask], probs[name], thr, "recall")
            m["recall_lo"], m["recall_hi"] = lo, hi
        rows.append(m)
    # flush: stdout is block-buffered when piped, so without this a multi-hour run
    # shows nothing until it finishes and a crash gives no clue how far it got
    print(f"  {arm:<18} done in {(time.time()-t0)/60:.1f} min  "
          f"test recall {rows[-1]['recall']:.3f}  spec {rows[-1]['specificity']:.3f}  "
          f"auc {rows[-1]['auc']:.4f}", flush=True)

    np.save(os.path.join(RESULTS, f"probs_{arm}_test.npy"), probs["test"])
    return pd.DataFrame(rows), clf, names


def train_ann(Xtr, ytr, Xva, yva, Xte, n_trials=10, seed=M.SEED):
    """Small MLP tuned with Optuna. Runs on the GPU; the input is already dense."""
    import optuna, torch
    from torch import nn

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tXtr = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    tytr = torch.tensor(ytr, dtype=torch.float32, device=dev)
    tXva = torch.tensor(Xva, dtype=torch.float32, device=dev)
    tXte = torch.tensor(Xte, dtype=torch.float32, device=dev)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                              dtype=torch.float32, device=dev)

    def build(hidden, depth, dropout):
        layers, d = [], Xtr.shape[1]
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        return nn.Sequential(*layers, nn.Linear(d, 1)).to(dev)

    def fit(hidden, depth, dropout, lr,
            epochs=M.CFG["ann"]["epochs"], bs=M.CFG["ann"]["batch_size"]):
        torch.manual_seed(seed)
        net = build(hidden, depth, dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=lr)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        n = len(tXtr)
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                lossf(net(tXtr[idx]).squeeze(1), tytr[idx]).backward()
                opt.step()
        net.eval()
        return net

    def predict(net, X):
        with torch.no_grad():
            return torch.sigmoid(net(X).squeeze(1)).cpu().numpy()

    from sklearn.metrics import roc_auc_score

    def objective(trial):
        sp = M.CFG["ann"]["search"]
        net = fit(trial.suggest_categorical("hidden", sp["hidden"]),
                  trial.suggest_int("depth", *sp["depth"]),
                  trial.suggest_float("dropout", *sp["dropout"]),
                  trial.suggest_float("lr", *sp["lr"], log=True))
        return roc_auc_score(yva, predict(net, tXva))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    net = fit(**study.best_params)
    return {"train": predict(net, tXtr), "val": predict(net, tXva), "test": predict(net, tXte)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS, choices=ARMS)
    ap.add_argument("--init-points", type=int, default=M.CFG["bayes_opt"]["init_points"])
    ap.add_argument("--n-iter", type=int, default=M.CFG["bayes_opt"]["n_iter"])
    ap.add_argument("--smoke", action="store_true", help="a couple of BO iterations, just check it runs")
    a = ap.parse_args()
    if a.smoke:
        a.init_points = a.n_iter = M.CFG["bayes_opt"]["smoke"]

    os.makedirs(RESULTS, exist_ok=True)
    df = M.load()
    split = M.make_splits(df)
    M.describe_splits(df, split)
    print(f"\nrunning arms: {', '.join(a.arms)}  (BO {a.init_points}+{a.n_iter})\n")

    all_rows, keep = [], {}
    for arm in a.arms:
        res, clf, names = run_arm(arm, df, split, a.init_points, a.n_iter)
        all_rows.append(res)
        if clf is not None:
            keep[arm] = (clf, names)

    out = pd.concat(all_rows, ignore_index=True)
    path = os.path.join(RESULTS, "arm_metrics.csv")
    if os.path.exists(path) and set(a.arms) != set(ARMS):
        old = pd.read_csv(path)
        out = pd.concat([old[~old["arm"].isin(a.arms)], out], ignore_index=True)
    out.to_csv(path, index=False)

    # keep the fitted model for the SHAP pass
    import pickle
    for arm, (clf, names) in keep.items():
        with open(os.path.join(RESULTS, f"model_{arm}.pkl"), "wb") as fh:
            pickle.dump({"model": clf, "feature_names": names}, fh)

    print(f"\nwrote {path}")
    test = out[out["split"] == "test"].set_index("arm")
    print("\nTEST SET COMPARISON")
    print(test[["auc", "recall", "specificity", "precision", "accuracy", "n"]].round(4).to_string())


if __name__ == "__main__":
    main()
