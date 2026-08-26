"""Train a dedicated bend-vs-non-bend OTDR detector using dense sliding windows.

This script uses only Dataset_event_stratified_split/train and /val.
It does NOT open Dataset_event_stratified_split/test.

Output:
- validation threshold sweep for bend detection
- selected bend model bundle
- CSV reports and a threshold-vs-metrics figure
"""

from pathlib import Path
import json
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "Dataset_event_stratified_split"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
OUT_DIR = APP_DIR / "gt_event_model_data_4class" / "bend_detector_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BEND_TOLERANCE_M = 5.0
POSITIVE_RADIUS_M = 2.0
GRID_STEP_M = 1.0
NMS_RADIUS_M = 5.0
MAX_POSITIVE_ROWS_PER_BEND = 5
MAX_NEGATIVE_ROWS_PER_TRACE = 35

FEATURE_COLS = [
    "m_norm", "db_at_event", "local_mean_db", "local_std_db",
    "pre_mean_db", "post_mean_db", "loss_dB", "peak_above_bg_dB",
    "pre_slope_dB_per_km", "post_slope_dB_per_km", "slope_change_dB_per_km",
    "derivative_at_event_dB_per_km", "max_pre_derivative", "min_post_derivative",
    "pre_std_db", "post_std_db", "post_to_pre_std_ratio", "peak_width_m",
    "local_range_db",
]

sys.path.append(str(APP_DIR))
try:
    from otdr_common import parse_otdr_csv, trim_dead_zone
except ImportError as error:
    raise ImportError("Place this script in the same folder as otdr_common.py.") from error


def m_to_samples(km, meters, minimum=2):
    if len(km) < 2:
        return minimum
    dx = float(np.median(np.diff(km)))
    return max(minimum, int((meters / 1000.0) / dx)) if dx > 0 else minimum


def safe_mean(values, fallback=0.0):
    return float(np.mean(values)) if len(values) else float(fallback)


def safe_std(values):
    return float(np.std(values)) if len(values) > 1 else 0.0


def local_slope(x, y):
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 3 and np.ptp(x) > 0 else 0.0


def peak_width(km, db, idx, bg, height):
    if height <= 0:
        return 0.0
    half_level = bg + height / 2.0
    left, right = idx, idx
    while left > 0 and db[left] > half_level:
        left -= 1
    while right < len(db) - 1 and db[right] > half_level:
        right += 1
    return float((km[right] - km[left]) * 1000.0)


def extract_features(km, db, idx):
    n = len(db)
    ws, wc, wsl = m_to_samples(km, 1.5), m_to_samples(km, 6.0), m_to_samples(km, 5.0)
    left = db[max(0, idx - wc):idx]
    right = db[idx + 1:min(n, idx + 1 + wc)]
    local = db[max(0, idx - ws):min(n, idx + ws + 1)]
    pre, post = safe_mean(left, db[idx]), safe_mean(right, db[idx])
    bg_values = np.concatenate([left, right]) if len(left) + len(right) else np.array([db[idx]])
    bg = float(np.median(bg_values))
    peak = float(np.max(local)) if len(local) else float(db[idx])
    grad = np.gradient(db, km) if n > 2 else np.zeros(n)
    pre_grad = grad[max(0, idx - ws):idx + 1]
    post_grad = grad[idx:min(n, idx + ws + 1)]
    nominal = float(km[-1] * 1000.0)
    pre_slope = local_slope(km[max(0, idx - wsl):idx], db[max(0, idx - wsl):idx])
    post_slope = local_slope(km[idx + 1:min(n, idx + 1 + wsl)], db[idx + 1:min(n, idx + 1 + wsl)])

    return {
        "m_norm": float(km[idx] * 1000.0 / nominal),
        "db_at_event": float(db[idx]),
        "local_mean_db": safe_mean(local, db[idx]),
        "local_std_db": safe_std(local),
        "pre_mean_db": pre,
        "post_mean_db": post,
        "loss_dB": float(pre - post),
        "peak_above_bg_dB": float(peak - bg),
        "pre_slope_dB_per_km": pre_slope,
        "post_slope_dB_per_km": post_slope,
        "slope_change_dB_per_km": post_slope - pre_slope,
        "derivative_at_event_dB_per_km": float(grad[idx]),
        "max_pre_derivative": float(np.max(pre_grad)) if len(pre_grad) else 0.0,
        "min_post_derivative": float(np.min(post_grad)) if len(post_grad) else 0.0,
        "pre_std_db": safe_std(left),
        "post_std_db": safe_std(right),
        "post_to_pre_std_ratio": float(safe_std(right) / (safe_std(left) + 1e-6)),
        "peak_width_m": peak_width(km, db, idx, bg, peak - bg),
        "local_range_db": float(np.ptp(local)) if len(local) else 0.0,
    }


def bend_positions(mask_path):
    with mask_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [float(a["m"]) for a in data.get("annotations", []) if str(a.get("type", "")).lower() == "bend"]


def dense_indices(km):
    guard = m_to_samples(km, 5.0)
    stride = max(1, m_to_samples(km, GRID_STEP_M))
    return list(range(guard, len(km) - guard, stride))


def make_dataset(folder, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for csv_path in sorted(folder.glob("*.csv")):
        mask_path = csv_path.with_suffix(".mask.json")
        if not mask_path.exists():
            continue
        try:
            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)
            bends = np.array(bend_positions(mask_path), dtype=float)
            indices = dense_indices(km)
            meters = np.array([km[idx] * 1000.0 for idx in indices])

            positive_indices = []
            negative_indices = []
            for idx, meter in zip(indices, meters):
                if len(bends) and np.min(np.abs(bends - meter)) <= POSITIVE_RADIUS_M:
                    positive_indices.append(idx)
                elif len(bends) == 0 or np.min(np.abs(bends - meter)) > BEND_TOLERANCE_M:
                    negative_indices.append(idx)

            if len(positive_indices) > MAX_POSITIVE_ROWS_PER_BEND * max(1, len(bends)):
                positive_indices = list(rng.choice(positive_indices, MAX_POSITIVE_ROWS_PER_BEND * len(bends), replace=False))
            if len(negative_indices) > MAX_NEGATIVE_ROWS_PER_TRACE:
                negative_indices = list(rng.choice(negative_indices, MAX_NEGATIVE_ROWS_PER_TRACE, replace=False))

            for idx in positive_indices:
                rows.append({"file": csv_path.name, "m": float(km[idx] * 1000.0), "label": "bend", **extract_features(km, db, idx)})
            for idx in negative_indices:
                rows.append({"file": csv_path.name, "m": float(km[idx] * 1000.0), "label": "non_bend", **extract_features(km, db, idx)})
        except Exception as error:
            print(f"[Skipped] {csv_path.name}: {error}")
    return pd.DataFrame(rows)


def nms(predictions):
    selected = []
    for event in sorted(predictions, key=lambda x: x["confidence"], reverse=True):
        if all(abs(event["m"] - kept["m"]) > NMS_RADIUS_M for kept in selected):
            selected.append(event)
    return sorted(selected, key=lambda x: x["m"])


def validation_predictions(model, folder, threshold):
    records = []

    for csv_path in sorted(folder.glob("*.csv")):
        mask_path = csv_path.with_suffix(".mask.json")

        if not mask_path.exists():
            continue

        try:
            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)

            idxs = dense_indices(km)

            if not idxs:
                print(f"[Skipped validation] {csv_path.name}: no valid dense-window positions")
                continue

            data = [extract_features(km, db, idx) for idx in idxs]
            X = pd.DataFrame(data)[FEATURE_COLS]

            probabilities = model.predict_proba(X)
            bend_column = list(model.classes_).index("bend")

            events = [
                {
                    "m": float(km[idx] * 1000.0),
                    "confidence": float(probabilities[row_i, bend_column]),
                }
                for row_i, idx in enumerate(idxs)
                if probabilities[row_i, bend_column] >= threshold
            ]

            records.append({
                "file": csv_path.name,
                "predictions": nms(events),
                "gt": bend_positions(mask_path),
            })

        except Exception as error:
            print(f"[Skipped validation] {csv_path.name}: {error}")

    return records


def event_metrics(records):
    tp = fp = fn = 0
    for record in records:
        gt = record["gt"]
        pred = record["predictions"]
        pairs = sorted((abs(g - p["m"]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred) if abs(g - p["m"]) <= BEND_TOLERANCE_M)
        used_g, used_p = set(), set()
        for _, gi, pi in pairs:
            if gi not in used_g and pi not in used_p:
                used_g.add(gi); used_p.add(pi)
        tp += len(used_g)
        fn += len(gt) - len(used_g)
        fp += len(pred) - len(used_p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    print("Building dense-window bend datasets from train and validation masks...")
    train_df = make_dataset(TRAIN_DIR, RANDOM_STATE)
    val_df = make_dataset(VAL_DIR, RANDOM_STATE + 1)
    train_df.to_csv(OUT_DIR / "bend_train_windows.csv", index=False)
    val_df.to_csv(OUT_DIR / "bend_val_windows.csv", index=False)

    print("\nTrain windows:")
    print(train_df["label"].value_counts())
    print("Validation windows:")
    print(val_df["label"].value_counts())
    assert set(train_df["file"]).isdisjoint(set(val_df["file"])), "File leakage detected."

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    models = {
        "ExtraTrees": ExtraTreesClassifier(n_estimators=800, max_depth=18, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=800, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=350, max_leaf_nodes=20, l2_regularization=1.0, random_state=RANDOM_STATE),
        "LogisticRegression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE))]),
    }

    thresholds = np.arange(0.10, 0.91, 0.05)
    all_results, fitted = [], {}
    for name, model in models.items():
        print(f"Training {name}...")
        fitted_model = clone(model).fit(X_train, y_train)
        fitted[name] = fitted_model
        for threshold in thresholds:
            metrics = event_metrics(validation_predictions(fitted_model, VAL_DIR, float(threshold)))
            all_results.append({"model": name, "threshold": float(threshold), **metrics})

    results = pd.DataFrame(all_results).sort_values(["f1", "recall", "precision"], ascending=False).reset_index(drop=True)
    results.to_csv(OUT_DIR / "bend_validation_threshold_sweep.csv", index=False)
    print("\nTop validation configurations:")
    print(results.head(15).round(4).to_string(index=False))

    best = results.iloc[0]
    best_name, best_threshold = best["model"], float(best["threshold"])
    best_model = fitted[best_name]
    print(f"\nSelected bend model: {best_name}; threshold={best_threshold:.2f}; validation bend F1={best['f1']:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in results["model"].unique():
        subset = results[results["model"] == name].sort_values("threshold")
        ax.plot(subset["threshold"], subset["f1"], marker="o", label=name)
    ax.axvline(best_threshold, color="black", ls="--", lw=1, label=f"selected threshold ({best_threshold:.2f})")
    ax.set_xlabel("Bend confidence threshold")
    ax.set_ylabel("Validation event-level F1")
    ax.set_title("Bend detector threshold selection on validation traces")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bend_validation_threshold_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURE_COLS,
        "positive_class": "bend",
        "threshold": best_threshold,
        "grid_step_m": GRID_STEP_M,
        "nms_radius_m": NMS_RADIUS_M,
        "match_tolerance_m": BEND_TOLERANCE_M,
        "selection_metric": "validation end-to-end bend event F1",
        "validation_metrics": {key: float(best[key]) for key in ["TP", "FP", "FN", "precision", "recall", "f1"]},
    }
    joblib.dump(bundle, OUT_DIR / "best_bend_detector.joblib")
    print(f"Saved bend model: {OUT_DIR / 'best_bend_detector.joblib'}")