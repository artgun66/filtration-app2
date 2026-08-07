"""Replace the 3B scam-type LLM with a head small enough to ship on a phone.

Stage 2 used Qwen2.5-3B because it wrote prose explanations. The app needs only the
type, and a 13-way label does not justify 2 GB of weights and 2.9 s per message. This
fits a linear head on the representation stage 1 already computes, so the app carries
one encoder and two heads.

  python distill.py --train                    # fit + evaluate, minutes
  python distill.py --compare                  # corrected labels vs the raw mapping
  python distill.py --vs-teacher               # the head vs the 3B it replaces
  python distill.py --pseudo-label --limit 4000  # extend coverage with the 3B, hours
  python distill.py --train --with-pseudo

What limits this is labels, not modelling. Of 29,653 scam rows only 1,055 carry a
type, and they come from SmishTank's own categories:

  delivery and toll 180, bank alert 116, tech support 102, government impersonation 85,
  romance 65, investment and crypto 64, prize and lottery 57, job offer 25,
  utility shutoff 1, other 360

`labeling/categorized_261.csv` corrects 136 of those, which is why the trainable pool
is 764 rows rather than 695. Two thirds of the gain is rows rescued out of `other`:

                       mapping   corrected     both scored on the same held-out rows
  accuracy               0.775       0.838
  macro-F1               0.710       0.784

`other` itself stays excluded. Per labeling/README.md it is 262 genuine
advertisement/loan rows plus 98 that fell through the brand map -- 27% contaminated --
so it teaches the head to reproduce a known bug. Low-confidence predictions return
None instead, which is the same answer without the false precision.

Four of the plan's thirteen types are deferred rather than learnt: charity and family
emergency have no rows anywhere in the corpus, Medicare/health and utility shutoff have
three each. A classifier cannot learn a class it has never seen, so the head covers
eight types and returns None elsewhere rather than guessing. Only labeling/'s 79
NEEDS_SOURCING work orders can change that -- no amount of relabelling will.
"""
import argparse, os, pickle, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scam-classification"))
# the 3B teacher stays in the research tree; only the distilled head belongs here
sys.path.insert(0, os.path.join(ROOT, "scam-type-classification"))

import modeling as M                            # noqa: E402
import run_arms as R                            # noqa: E402

RESULTS = os.path.join(HERE, "results")
PSEUDO = os.path.join(RESULTS, "pseudo_labels.csv")
CORRECTED = os.path.join(ROOT, "labeling", "categorized_261.csv")
EXCLUDE = {"other"}          # contaminated bucket, see the module docstring
MIN_PER_CLASS = 10           # below this a class cannot be evaluated, let alone learnt

# The four types the corpus cannot support: charity and family emergency have no rows
# at all, Medicare and utility shutoff have 3 and 3. Deferred by decision rather than
# left to MIN_PER_CLASS, so that pseudo-labelling cannot quietly promote them later on
# teacher output alone. labeling/'s 79 NEEDS_SOURCING work orders are what lifts them.
DEFERRED = {"family emergency", "charity", "medicare and health", "utility shutoff"}


def head_path(arm):
    return os.path.join(M.RESULTS, f"type_head_{arm}.pkl")


# ---------------------------------------------------------------- training data
def corrected_types():
    """id -> type from labeling/categorized_261.csv, or {} when it has not been built.

    See labeling/extract_categorized.py for what these labels are and are not. They
    correct the mapping on 52% of the rows they touch, including all 98 that the brand
    fallback dumped into `other`, but they are classifier-assigned rather than blind
    annotation -- so they are a better training signal, not ground truth.
    """
    if not os.path.exists(CORRECTED):
        return {}
    c = pd.read_csv(CORRECTED)
    return dict(zip(c["id"].astype(str), c["category"].astype(str)))


def usable(frame, quiet=False):
    """Drop what cannot be trained on: the excluded buckets, then the thin classes."""
    frame = frame[~frame["type"].isin(EXCLUDE | DEFERRED)]
    counts = frame["type"].value_counts()
    thin = counts[counts < MIN_PER_CLASS]
    if len(thin) and not quiet:
        print(f"dropping {len(thin)} class(es) under {MIN_PER_CLASS} rows: "
              + ", ".join(f"{t} ({n})" for t, n in thin.items()))
    return frame[frame["type"].isin(counts[counts >= MIN_PER_CLASS].index)]


def training_rows(df, with_pseudo=False, corrected=True, quiet=False):
    """(ids, labels, provenance) for every row that carries a usable type.

    Corrections are applied *before* the `other` exclusion, which is the whole point:
    76 of the rows they rescue are sitting in `other` and would otherwise be dropped
    before they could be relabelled.
    """
    frame = pd.DataFrame({"id": df["id"].to_numpy(),
                          "type": df["scam_type"].astype(str).to_numpy(),
                          "src": "mapped"})

    if corrected:
        fix = frame["id"].astype(str).map(corrected_types())
        changed = fix.notna() & (fix != frame["type"])
        frame.loc[changed, "src"] = "corrected"
        frame["type"] = fix.fillna(frame["type"])
        if not quiet:
            print(f"{fix.notna().sum()} rows carry a corrected label, "
                  f"{changed.sum()} of them differing from the mapping")

    frame = frame[frame["type"] != ""]

    if with_pseudo:
        if not os.path.exists(PSEUDO):
            raise SystemExit(f"no {PSEUDO} -- run: python distill.py --pseudo-label")
        ps = pd.read_csv(PSEUDO)
        ps = ps[(~ps["type"].isin(EXCLUDE)) & (~ps["id"].isin(frame["id"]))]
        frame = pd.concat([frame, pd.DataFrame({"id": ps["id"], "type": ps["type"],
                                                "src": "pseudo"})], ignore_index=True)

    return usable(frame, quiet)


def _fit(Xtr, ytr, seed):
    from sklearn.linear_model import LogisticRegression

    # Linear, not boosted: ~700 rows against 400+ dimensions is where a tree ensemble
    # memorises. class_weight balances delivery-and-toll (186) against job offer (26).
    # 5000, not 2000: the engineered features are unscaled next to unit-norm embeddings,
    # so lbfgs needs ~2800 iterations and was previously stopping short of the optimum.
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced",
                             random_state=seed)
    clf.fit(Xtr, ytr)
    return clf


def _rows_to_X(df, arm, frame):
    """The stage-1 representation for `frame`, from the same cache the app uses."""
    X_all, names = R.feature_matrix(arm, df, df["id"].to_numpy())
    pos = {int(i): k for k, i in enumerate(df["id"].to_numpy())}
    return X_all[[pos[int(i)] for i in frame["id"]]], names


def train(arm, with_pseudo=False, seed=M.SEED, corrected=True):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    df = M.load()
    frame = training_rows(df, with_pseudo, corrected)
    print(f"\n{len(frame)} labelled rows over {frame['type'].nunique()} types "
          f"({(frame['src'] == 'pseudo').sum()} pseudo, "
          f"{(frame['src'] == 'corrected').sum()} corrected)")
    print(frame["type"].value_counts().to_string())

    # Same representation as stage 1, from the same cache -- the app computes this
    # vector once and feeds both heads.
    X, names = _rows_to_X(df, arm, frame)
    y = frame["type"].to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed,
                                          stratify=y)
    clf = _fit(Xtr, ytr, seed)

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
    if corrected:
        print("NOTE: scored against labeling/categorized_261.csv where it has an "
              "opinion and the\n      mapping elsewhere. Those labels are "
              "classifier-assigned, so this measures\n      agreement with a better "
              "labeller, not ground truth. The blind pass in\n      labeling/README.md "
              "is still what settles it.")
    else:
        print("NOTE: scored against the mapped labels, which labeling/README.md "
              "documents as buggy.")
    return clf


TEACHER_EVAL = os.path.join(ROOT, "scam-type-classification", "results",
                            "scam_type_eval_choice.csv")


def _held_out(df, arm, seed):
    """The exact test split train() uses, so nothing is scored on its own fit."""
    from sklearn.model_selection import train_test_split

    frame = training_rows(df, quiet=True)
    tr, te = train_test_split(frame, test_size=0.25, random_state=seed,
                              stratify=frame["type"])
    return tr, te


def vs_teacher(arm, seed=M.SEED):
    """The 3B teacher against the distilled head, on the same rows and same labels.

    The published 0.742 was scored against the mapping, and the head is now scored
    against the corrections -- so the two numbers were never comparable. This rescores
    the teacher's saved predictions on the corrected labels and restricts both to the
    head's held-out split, which is the only set neither model has an unfair claim on.

    The teacher's eval CSV carries no id, so the join is on message text.
    """
    from sklearn.metrics import f1_score

    if not os.path.exists(TEACHER_EVAL):
        raise SystemExit(f"no {TEACHER_EVAL}")

    df = M.load()
    tr, te = _held_out(df, arm, seed)

    ev = pd.read_csv(TEACHER_EVAL)
    text_to_id = dict(zip(df["text"].astype(str), df["id"]))
    ev["id"] = ev["text"].astype(str).map(text_to_id)
    joined = ev["id"].notna().sum()
    print(f"teacher rows {len(ev)}, joined to the corpus by text: {joined}")

    truth = dict(zip(te["id"], te["type"]))
    ev = ev[ev["id"].isin(truth)].copy()
    if ev.empty:
        raise SystemExit("no teacher rows fall in the held-out split")
    # A message can appear more than once in the corpus; one prediction per id is enough.
    ev = ev.drop_duplicates(subset="id")
    ev["truth"] = ev["id"].map(truth)

    Xte, _ = _rows_to_X(df, arm, te)
    Xtr, _ = _rows_to_X(df, arm, tr)
    head = _fit(Xtr, tr["type"].to_numpy(), seed)
    head_pred = pd.Series(head.predict(Xte), index=te["id"].to_numpy())

    ev["head"] = ev["id"].map(head_pred)
    print(f"scored on {len(ev)} held-out rows both models answered\n")

    # macro-F1 is averaged over the head's 8 classes for both models. Left unrestricted
    # it averages over every label either model emits, so the teacher is charged an F1
    # of 0 for each of the 5 types with no rows in this split -- which measures the
    # taxonomy, not the model. Answers outside the 8 still count against it, as misses.
    classes = sorted(set(ev["truth"]))
    for name, col in [("3B teacher", "pred"), ("linear head", "head")]:
        acc = (ev[col] == ev["truth"]).mean()
        f1 = f1_score(ev["truth"], ev[col], labels=classes, average="macro",
                      zero_division=0)
        outside = (~ev[col].isin(classes)).sum()
        print(f"  {name:<12} accuracy {acc:.3f}   macro-F1 {f1:.3f}"
              f"   answered outside the 8: {outside}")


def compare(arm, seed=M.SEED):
    """Corrected labels against the mapping, scored on the same held-out rows.

    Training on different labels also changes which rows are trainable at all -- the
    corrections rescue 76 from `other` -- so the two heads see different training sets
    by design. What has to be held fixed is the test set: the same messages, judged
    against the best labels available, which is the only way the delta means anything.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    df = M.load()
    new = training_rows(df, corrected=True)
    prior = training_rows(df, corrected=False, quiet=True)
    old = dict(zip(prior["id"].astype(str), prior["type"]))

    tr, te = train_test_split(new, test_size=0.25, random_state=seed,
                              stratify=new["type"])
    Xtr, _ = _rows_to_X(df, arm, tr)
    Xte, _ = _rows_to_X(df, arm, te)
    yte = te["type"].to_numpy()

    # The baseline can only train on rows that had a usable mapped label, which is
    # precisely the handicap being measured -- so it trains on the subset, not on
    # corrected labels for rows it never would have had.
    keep = [i for i, r in enumerate(tr["id"].astype(str)) if r in old]
    base_y = np.array([old[r] for r in tr["id"].astype(str).iloc[keep]])

    print(f"\ntrain rows: {len(tr)} corrected, {len(keep)} available to the mapping")
    print(f"test rows:  {len(te)} (shared, judged on corrected labels)\n")

    for name, clf in [("mapping  ", _fit(Xtr[keep], base_y, seed)),
                      ("corrected", _fit(Xtr, tr["type"].to_numpy(), seed))]:
        pred = clf.predict(Xte)
        acc = (pred == yte).mean()
        rep = classification_report(yte, pred, zero_division=0, output_dict=True)
        print(f"{name}  accuracy {acc:.3f}   macro-F1 {rep['macro avg']['f1-score']:.3f}"
              f"   classes {len(clf.classes_)}")


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
    ap.add_argument("--no-corrections", action="store_true",
                    help="ignore labeling/categorized_261.csv, train on the mapping")
    ap.add_argument("--compare", action="store_true",
                    help="corrected labels vs the mapping on the same held-out rows")
    ap.add_argument("--vs-teacher", action="store_true",
                    help="the 3B teacher vs the head, same rows and same labels")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=0)
    a = ap.parse_args()

    if a.pseudo_label:
        pseudo_label(a.limit, a.batch_size or None)
    if a.compare:
        compare(a.arm)
    if a.vs_teacher:
        vs_teacher(a.arm)
    if a.train:
        train(a.arm, a.with_pseudo, corrected=not a.no_corrections)
    if not (a.train or a.pseudo_label or a.compare or a.vs_teacher):
        ap.error("pass --train, --compare, --vs-teacher or --pseudo-label")


if __name__ == "__main__":
    main()
