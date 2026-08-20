"""
train_models.py — ТОЛЬКО обучение + сравнение моделей на уже готовых
train_events.csv / val_events.csv (созданных split_dataset.py).

Обучает 6 разных классификаторов на train, сравнивает их на held-out val
(val не участвует в обучении/подборе гиперпараметров), выбирает лучшую
по val_f1_macro и сохраняет её.

Запуск:
    python train_models.py --train event_model_results/train_events.csv --val event_model_results/val_events.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

FEATURE_COLS = ["loss_dB", "reflectance_dB", "is_reflective", "width_m",
                "frac_of_expected", "dist_to_eof_m", "post_signal_std",
                "post_signal_mean_db", "pre_slope_db_per_km", "post_slope_db_per_km"]

MODELS = {
    "DecisionTree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                           class_weight="balanced", random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=300, max_depth=8,
                                       class_weight="balanced", random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
    ]),
}


def main():
    ap = argparse.ArgumentParser(description="Обучение + сравнение моделей на train/val CSV.")
    ap.add_argument("--train", required=True, help="путь к train_events.csv")
    ap.add_argument("--val", required=True, help="путь к val_events.csv")
    ap.add_argument("--out", default="event_model_results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    train_df = pd.read_csv(args.train).dropna(subset=FEATURE_COLS)
    val_df = pd.read_csv(args.val).dropna(subset=FEATURE_COLS)

    Xtr, ytr = train_df[FEATURE_COLS].values, train_df["label"].values
    Xva, yva = val_df[FEATURE_COLS].values, val_df["label"].values

    print(f"Train rows: {len(Xtr)} | class counts:\n{train_df['label'].value_counts()}\n")
    print(f"Val rows:   {len(Xva)} | class counts:\n{val_df['label'].value_counts()}\n")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results = []
    fitted = {}

    for name, model in MODELS.items():
        try:
            cv_scores = cross_val_score(model, Xtr, ytr, cv=cv, scoring="f1_macro")
        except Exception as e:
            print(f"[skip cv] {name}: {e}")
            cv_scores = np.array([np.nan])

        model.fit(Xtr, ytr)
        pred_val = model.predict(Xva)

        acc = accuracy_score(yva, pred_val)
        p, r, f1, _ = precision_recall_fscore_support(yva, pred_val, average="macro", zero_division=0)
        pw, rw, f1w, _ = precision_recall_fscore_support(yva, pred_val, average="weighted", zero_division=0)

        results.append({
            "model": name,
            "cv_f1_macro_train": round(float(cv_scores.mean()), 4),
            "cv_f1_macro_std": round(float(cv_scores.std()), 4),
            "val_accuracy": round(acc, 4),
            "val_f1_macro": round(f1, 4),
            "val_precision_macro": round(p, 4),
            "val_recall_macro": round(r, 4),
            "val_f1_weighted": round(f1w, 4),
        })
        fitted[name] = model

        print(f"=== {name} ===")
        print(classification_report(yva, pred_val, digits=3, zero_division=0))
        labels_sorted = sorted(set(yva) | set(pred_val))
        cm = confusion_matrix(yva, pred_val, labels=labels_sorted)
        print("confusion (rows=true, cols=pred):", labels_sorted)
        print(cm, "\n")

    results_df = pd.DataFrame(results).sort_values("val_f1_macro", ascending=False)
    results_df.to_csv(os.path.join(args.out, "model_comparison.csv"), index=False)
    print("=== MODEL COMPARISON (sorted by val_f1_macro) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    classes = list(best_model.classes_) if hasattr(best_model, "classes_") \
        else list(best_model.named_steps["clf"].classes_)

    joblib.dump({"model": best_model, "features": FEATURE_COLS,
                 "classes": classes, "model_name": best_name},
                os.path.join(args.out, "event_clf_best.joblib"))
    print(f"\nBest model: {best_name} -> saved {os.path.join(args.out, 'event_clf_best.joblib')}")


if __name__ == "__main__":
    main()
