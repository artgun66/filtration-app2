"""Per-scam-type metrics from the corpus columns, written to CSV.

Reads `scam_type_true` and `scam_type_pred` as annotate.py --types wrote them, and
scores them one class at a time. One-vs-rest, so precision and recall mean the usual
thing for a single type: of the messages called `bank alert`, how many were; of the
messages that were, how many were called it.

    python type_metrics.py                 # held-out rows only
    python type_metrics.py --split all     # every row, in-sample and worthless
    python type_metrics.py --split test

`test` alone is the honest split but it is small -- most labelled rows are SmishTank,
which lands 60% in train. The default is val+test together for that reason, and the
`n` column is there so a class with six rows is not read as a measurement.

CAVEAT, and it is not small. The type head trained on 26% test-split rows, so even
`--split test` is not clean for stage 2. labeling/build_test_set.py exists to fix
exactly this: label those 188 rows, retrain without them, and this becomes a number
worth quoting.
"""
import argparse
import os

import pandas as pd

import modeling as M

RESULTS = M.RESULTS
OUT = os.path.join(RESULTS, "type_metrics.csv")

# Types the head is never trained on, so a labelled row of one of these can only ever
# be counted wrong. Kept in the `all rows` table for honesty and excluded from the
# `learnable only` one, which is the comparable number.
UNTRAINED = {"family emergency", "charity", "medicare and health", "utility shutoff"}


def metrics(true, pred, labels):
    """One-vs-rest precision/recall/F1 per label, plus the macro and weighted rows."""
    rows = []
    for c in labels:
        tp = int(((true == c) & (pred == c)).sum())
        fp = int(((true != c) & (pred == c)).sum())
        fn = int(((true == c) & (pred != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"scam_type": c, "n_true": tp + fn, "n_pred": tp + fp,
                     "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4)})

    df = pd.DataFrame(rows)
    support = df["n_true"].sum()
    for name, weights in [("MACRO AVG", None), ("WEIGHTED AVG", df["n_true"])]:
        w = None if weights is None else (weights / support if support else weights * 0)
        agg = {"scam_type": name, "n_true": int(support),
               "n_pred": int(df["n_pred"].sum()),
               "tp": int(df["tp"].sum()), "fp": int(df["fp"].sum()),
               "fn": int(df["fn"].sum())}
        for col in ("precision", "recall", "f1"):
            agg[col] = round(float((df[col] * w).sum() if w is not None
                                   else df[col].mean()), 4)
        df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    return df


def main(split="heldout"):
    df = pd.read_csv(M.DATASET_CSV, keep_default_na=False, dtype=str)
    for col in ("scam_type_true", "scam_type_pred"):
        if col not in df.columns:
            raise SystemExit(f"no {col} column -- run: "
                             f"python annotate.py --arm minilm_feat --types")

    if split == "heldout":
        df = df[df["split"].isin(["val", "test"])]
    elif split != "all":
        df = df[df["split"] == split]

    # Only rows with a label AND a prediction can be scored: an unflagged row has no
    # prediction, and stage 2 never runs on it.
    scored = df[(df["scam_type_true"] != "") & (df["scam_type_pred"] != "")]
    missed = df[(df["scam_type_true"] != "") & (df["scam_type_pred"] == "")]
    print(f"split={split}: {len(scored)} labelled rows stage 1 flagged, "
          f"{len(missed)} labelled rows it missed (no prediction to score)")

    labels = sorted(set(scored["scam_type_true"]) | set(scored["scam_type_pred"]))
    out = metrics(scored["scam_type_true"], scored["scam_type_pred"], labels)
    out.insert(0, "split", split)
    out.insert(1, "scope", "all rows")

    # The averages above are not a measurement of the head, and it is worth being
    # precise about why. `other` is excluded from training (27% contaminated, see
    # labeling/README.md) so the head can never predict it -- yet every labelled
    # `other` row still gets predicted as *something*. Those rows are pure false
    # positives spread across the real classes, which is what drives crypto to 0.42
    # precision at 1.00 recall. Same for the four deferred types. Scoring the classes
    # the head is actually allowed to answer is the comparable number.
    unlearnable = {"other"} | UNTRAINED
    keep = scored[~scored["scam_type_true"].isin(unlearnable)]
    dropped = len(scored) - len(keep)
    learnable = sorted(set(keep["scam_type_true"]) | set(keep["scam_type_pred"]))
    fair = metrics(keep["scam_type_true"], keep["scam_type_pred"], learnable)
    fair.insert(0, "split", split)
    fair.insert(1, "scope", "learnable only")

    print(f"\n--- all {len(scored)} labelled+flagged rows ---")
    print(out.to_string(index=False))
    print(f"\n--- {len(keep)} rows whose true type the head can predict "
          f"({dropped} dropped: `other` and the deferred types) ---")
    print(fair.to_string(index=False))
    out = pd.concat([out, fair], ignore_index=True)

    os.makedirs(RESULTS, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="heldout",
                   choices=["heldout", "test", "val", "train", "all"])
    main(**vars(p.parse_args()))
