"""
train_trend_clf.py — STAGE 3: supervised training + testing on trend features.

Flow:
  analyze_trends -> 7 trend features per trace
  -> stratified train/test split (75/25)
  -> 5-fold cross-validation on the train set (robust accuracy ± std)
  -> fit Decision Tree (interpretable) and Random Forest (stronger)
  -> held-out TEST report + confusion matrix for both
  -> feature importances + the decision-tree RULES (reads out your mental model)
  -> save the chosen model (+ feature list + expected length) for reuse.

Run:        python train_trend_clf.py
Predict:    from train_trend_clf import predict_file; predict_file("some.csv")
"""

import os, glob
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from otdr_common import label_from_filename
from otdr_trends import analyze_file
from otdr_cluster import trace_features, FEAT_COLS

# ---- config ---------------------------------------------------------------- #
ROOT       = r"C:\Users\PCA\Desktop\OTDR_CLassification\Dataset"
MODEL_PATH = r"C:\Users\PCA\Desktop\OTDR_CLassification\trend_clf.joblib"
TREE_PNG   = r"C:\Users\PCA\Desktop\OTDR_CLassification\trend_tree.png"
EXPECTED_M = 125.0          # empirical fibre length (normal traces end here). Tune to your cable.
NAMES      = {1: "normal", 2: "bend", 3: "break"}
LABELS     = [1, 2, 3]
TARGET     = [NAMES[i] for i in LABELS]
# ---------------------------------------------------------------------------- #


def build_dataset():
    rows = []
    for fp in glob.glob(os.path.join(ROOT, "*.csv")):
        lab = label_from_filename(fp)
        if lab is None:
            continue
        try:
            out = analyze_file(fp, expected_length_m=EXPECTED_M)
            f = trace_features(out, EXPECTED_M)
        except Exception as e:
            print(f"[skip] {os.path.basename(fp)}: {e}")
            continue
        f["file"] = os.path.basename(fp)
        f["y"] = lab + 1
        rows.append(f)
    df = pd.DataFrame(rows).fillna(0.0)
    print(f"{len(df)} traces | class counts: "
          f"{ {NAMES[k]: int((df['y']==k).sum()) for k in LABELS} }")
    return df


def main():
    df = build_dataset()
    X, y = df[FEAT_COLS].values, df["y"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    print(f"train {len(ytr)} | test {len(yte)}\n")

    dt = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)

    # cross-validation on the training set
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for name, clf in [("Decision Tree", dt), ("Random Forest", rf)]:
        s = cross_val_score(clf, Xtr, ytr, cv=cv)
        print(f"{name:14s} 5-fold CV acc: {s.mean():.3f} ± {s.std():.3f}")
    print()

    dt.fit(Xtr, ytr)
    rf.fit(Xtr, ytr)

    # held-out test
    for name, clf in [("Decision Tree", dt), ("Random Forest", rf)]:
        pred = clf.predict(Xte)
        print(f"=== {name} — TEST ===")
        print(classification_report(yte, pred, labels=LABELS, target_names=TARGET,
                                    digits=3, zero_division=0))
        print("confusion (rows=true, cols=pred) [normal, bend, break]:")
        print(confusion_matrix(yte, pred, labels=LABELS), "\n")

    # interpretability
    print("Random Forest feature importances:")
    for f, v in sorted(zip(FEAT_COLS, rf.feature_importances_), key=lambda t: -t[1]):
        print(f"   {f:20s} {v:.3f}")
    print("\nDecision-tree rules:")
    print(export_text(dt, feature_names=list(FEAT_COLS)))

    plt.figure(figsize=(18, 10))
    plot_tree(dt, feature_names=list(FEAT_COLS), class_names=TARGET, filled=True, fontsize=8)
    plt.tight_layout(); plt.savefig(TREE_PNG, dpi=130)
    print(f"saved tree diagram -> {TREE_PNG}")

    # save the stronger model + config
    joblib.dump({"model": rf, "features": list(FEAT_COLS),
                 "expected_m": EXPECTED_M, "names": NAMES}, MODEL_PATH)
    print(f"saved model -> {MODEL_PATH}")


def predict_file(path, model_path=MODEL_PATH):
    """Classify one new OTDR file with the trained model."""
    b = joblib.load(model_path)
    out = analyze_file(path, expected_length_m=b["expected_m"])
    f = trace_features(out, b["expected_m"])
    x = np.array([[f[c] for c in b["features"]]])
    cls = int(b["model"].predict(x)[0])
    proba = b["model"].predict_proba(x)[0]
    return {"file": os.path.basename(path),
            "class": cls, "class_name": b["names"][cls],
            "proba": {b["names"][c]: round(float(p), 3)
                      for c, p in zip(b["model"].classes_, proba)},
            "features": f}


if __name__ == "__main__":
    main()