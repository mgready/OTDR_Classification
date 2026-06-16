"""
batch_channel.py — run the trend-channel classifier across the whole dataset.

Compares the RULE-BASED trend verdict against the class1/2/3 filename labels, prints
a confusion matrix + report, lists every miss WITH the reason the rule fired, and dumps
the trend features of every file to a CSV. That CSV is the bridge to the "learned"
version: train a small classifier on it instead of the hand-set thresholds.

Run:  python batch_channel.py
"""

import os, glob, csv
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

from otdr_common import label_from_filename
from otdr_channel import analyze_file

# ---- config ---------------------------------------------------------------- #
ROOT             = r"C:\Users\PCA\Desktop\OTDR_CLassification\Dataset"
FEATURE_CSV      = r"C:\Users\PCA\Desktop\OTDR_CLassification\trend_features.csv"
EXPECTED_LEN_M   = 150.0
SIGMA_MULT       = 2.5
NAMES            = {1: "normal", 2: "bend", 3: "break"}     # 1-indexed to match prefixes
# ---------------------------------------------------------------------------- #


def true_class_from_name(path):
    """class1_*.csv -> 1, class2_* -> 2, class3_* -> 3 (None if no match)."""
    lab0 = label_from_filename(path)          # 0/1/2 or None
    return None if lab0 is None else lab0 + 1


def main():
    files = glob.glob(os.path.join(ROOT, "*.csv"))
    gts, preds, rows, feat_rows = [], [], [], []

    for fp in files:
        t = true_class_from_name(fp)
        if t is None:
            continue
        try:
            out = analyze_file(fp, expected_length_m=EXPECTED_LEN_M, sigma_mult=SIGMA_MULT)
        except Exception as e:
            print(f"[skip] {os.path.basename(fp)}: {e}")
            continue
        p = out["result"]["class"]
        gts.append(t); preds.append(p)
        rows.append((os.path.basename(fp), t, p, out["result"]["why"]))
        feat_rows.append({"file": os.path.basename(fp), "true_class": t,
                          "pred_class": p, **out["features"]})

    if not gts:
        raise RuntimeError("No files classified — check ROOT path.")

    print(f"classified {len(gts)} files | true counts: "
          f"{ {NAMES[k]: v for k, v in sorted(Counter(gts).items())} }\n")

    labels = [1, 2, 3]
    target = [NAMES[i] for i in labels]
    print("=== trend-channel rule classifier ===")
    print(classification_report(gts, preds, labels=labels,
                                target_names=target, digits=3, zero_division=0))
    print("confusion matrix (rows=true, cols=pred) order [normal, bend, break]:")
    print(confusion_matrix(gts, preds, labels=labels), "\n")

    wrong = [r for r in rows if r[1] != r[2]]
    print(f"{len(wrong)} misclassified (file | true -> pred | reason):")
    for nm, t, p, why in wrong[:60]:
        print(f"   {nm:42s} {NAMES[t]:6s} -> {NAMES[p]:6s} | {why}")

    # dump features for the learned version
    if feat_rows:
        keys = list(feat_rows[0].keys())
        with open(FEATURE_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(feat_rows)
        print(f"\nfeatures of all files written to: {FEATURE_CSV}")
        print("   -> train a classifier on columns "
              "[slope_dB_per_km, valid_fraction, step_drop_dB, ...] vs true_class")


if __name__ == "__main__":
    main()