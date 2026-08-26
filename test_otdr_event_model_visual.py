"""Fast final blind test for the hybrid OTDR detector.

Optimizations versus the visual version:
- batch feature prediction per trace instead of one model call per candidate;
- no plotting during inference;
- prints progress for every test trace;
- test masks are read only after all CSV-only predictions are written.

Run visualisation separately after metrics are produced.
"""

from pathlib import Path
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", message=".*sklearn configuration.*joblib workers.*")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

APP_DIR = Path(__file__).resolve().parent
TEST_DIR = APP_DIR / "Dataset_event_stratified_split" / "test"
SOURCE_DATASET = APP_DIR / "Dataset"
MAIN_MODEL_PATH = APP_DIR / "gt_event_model_data_4class" / "model_results_event_split" / "best_4class_event_classifier.joblib"
BEND_MODEL_PATH = APP_DIR / "gt_event_model_data_4class" / "bend_detector_results" / "best_bend_detector.joblib"
OUT_DIR = APP_DIR / "gt_event_model_data_4class" / "hybrid_test_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_CLASSES = ["bend", "connector", "break"]
MAIN_CONFIDENCE = 0.45
MATCH_TOLERANCE_M = 5.0
MAIN_NMS_RADIUS_M = 5.0

sys.path.append(str(APP_DIR))
from otdr_common import parse_otdr_csv, trim_dead_zone


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
    half = bg + height / 2.0
    left, right = idx, idx
    while left > 0 and db[left] > half:
        left -= 1
    while right < len(db) - 1 and db[right] > half:
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
    pre_slope = local_slope(km[max(0, idx - wsl):idx], db[max(0, idx - wsl):idx])
    post_slope = local_slope(km[idx + 1:min(n, idx + 1 + wsl)], db[idx + 1:min(n, idx + 1 + wsl)])
    nominal = float(km[-1] * 1000.0)
    return {
        "m_norm": float(km[idx] * 1000.0 / nominal), "db_at_event": float(db[idx]),
        "local_mean_db": safe_mean(local, db[idx]), "local_std_db": safe_std(local),
        "pre_mean_db": pre, "post_mean_db": post, "loss_dB": float(pre - post),
        "peak_above_bg_dB": float(peak - bg),
        "pre_slope_dB_per_km": pre_slope, "post_slope_dB_per_km": post_slope,
        "slope_change_dB_per_km": post_slope - pre_slope,
        "derivative_at_event_dB_per_km": float(grad[idx]),
        "max_pre_derivative": float(np.max(pre_grad)) if len(pre_grad) else 0.0,
        "min_post_derivative": float(np.min(post_grad)) if len(post_grad) else 0.0,
        "pre_std_db": safe_std(left), "post_std_db": safe_std(right),
        "post_to_pre_std_ratio": float(safe_std(right) / (safe_std(left) + 1e-6)),
        "peak_width_m": peak_width(km, db, idx, bg, peak - bg),
        "local_range_db": float(np.ptp(local)) if len(local) else 0.0,
    }


def moving_average(values, samples):
    samples = max(3, int(samples) | 1)
    return np.convolve(values, np.ones(samples) / samples, mode="same") if len(values) >= samples else values.astype(float)


def main_candidates(km, db):
    n = len(db)
    smoothed = moving_average(db, m_to_samples(km, 1.0))
    distance = m_to_samples(km, 1.5)
    mad = np.median(np.abs(smoothed - np.median(smoothed))) + 1e-6
    peaks, _ = find_peaks(smoothed, prominence=max(0.7, 1.5 * mad), distance=distance)
    step = np.zeros(n)
    for i in range(distance, n - distance):
        step[i] = np.median(smoothed[i - distance:i]) - np.median(smoothed[i:i + distance])
    threshold = max(0.25, np.percentile(step[distance:n - distance], 85)) if n > 2 * distance else 0.25
    drops, _ = find_peaks(step, height=threshold, distance=distance)
    minima, _ = find_peaks(-smoothed, prominence=max(0.25, 0.5 * mad), distance=distance)
    guard = m_to_samples(km, 5.0)
    return sorted({int(i) for i in np.concatenate([peaks, drops, minima]) if guard <= i < n - guard})


def dense_indices(km, step_m):
    guard = m_to_samples(km, 5.0)
    stride = max(1, m_to_samples(km, step_m))
    return list(range(guard, len(km) - guard, stride))


def nms(events, radius_m):
    kept = []
    for event in sorted(events, key=lambda x: x["confidence"], reverse=True):
        if all(abs(event["m"] - other["m"]) > radius_m for other in kept):
            kept.append(event)
    return sorted(kept, key=lambda x: x["m"])


def predict_main(km, db, bundle):
    indices = main_candidates(km, db)
    if not indices:
        return []
    model, cols = bundle["model"], bundle["features"]
    X = pd.DataFrame([extract_features(km, db, idx) for idx in indices])[cols]
    labels = model.predict(X)
    probabilities = model.predict_proba(X)
    class_index = {str(label): i for i, label in enumerate(model.classes_)}
    events = []
    for idx, label, proba in zip(indices, labels, probabilities):
        label = str(label)
        confidence = float(proba[class_index[label]])
        if label in {"connector", "break"} and confidence >= MAIN_CONFIDENCE:
            events.append({"m": float(km[idx] * 1000.0), "type": label, "confidence": confidence, "branch": "main"})
    return nms(events, MAIN_NMS_RADIUS_M)


def predict_bends(km, db, bundle):
    indices = dense_indices(km, float(bundle["grid_step_m"]))
    if not indices:
        return []
    model, cols = bundle["model"], bundle["features"]
    X = pd.DataFrame([extract_features(km, db, idx) for idx in indices])[cols]
    bend_index = list(model.classes_).index("bend")
    probabilities = model.predict_proba(X)[:, bend_index]
    events = [
        {"m": float(km[idx] * 1000.0), "type": "bend", "confidence": float(prob), "branch": "bend"}
        for idx, prob in zip(indices, probabilities)
        if prob >= float(bundle["threshold"])
    ]
    return nms(events, float(bundle["nms_radius_m"]))


def load_gt(file_name):
    path = SOURCE_DATASET / Path(file_name).with_suffix(".mask.json")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [{"m": float(a["m"]), "type": str(a["type"]).lower()} for a in data.get("annotations", []) if str(a.get("type", "")).lower() in EVENT_CLASSES]


def match_events(gt, pred):
    candidates = sorted((abs(g["m"] - p["m"]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred) if g["type"] == p["type"] and abs(g["m"] - p["m"]) <= MATCH_TOLERANCE_M)
    used_g, used_p, matches = set(), set(), []
    for distance, gi, pi in candidates:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi); used_p.add(pi); matches.append((gt[gi], pred[pi], distance))
    return matches, [g for i, g in enumerate(gt) if i not in used_g], [p for i, p in enumerate(pred) if i not in used_p]


if __name__ == "__main__":
    main_bundle = joblib.load(MAIN_MODEL_PATH)
    bend_bundle = joblib.load(BEND_MODEL_PATH)
    csv_files = sorted(TEST_DIR.glob("*.csv"))
    cache, predictions, errors = {}, [], []

    print(f"PHASE A: CSV-only prediction on {len(csv_files)} test traces")
    for i, csv_path in enumerate(csv_files, 1):
        try:
            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)
            main_events = predict_main(km, db, main_bundle)
            bend_events = predict_bends(km, db, bend_bundle)
            events = sorted(main_events + bend_events, key=lambda event: event["m"])
            cache[csv_path.name] = events
            predictions.extend({"file": csv_path.name, **event} for event in events)
            print(f"[{i:02d}/{len(csv_files)}] {csv_path.name}: main={len(main_events)}, bend={len(bend_events)}")
        except Exception as error:
            errors.append({"file": csv_path.name, "stage": "prediction", "error": str(error)})
            print(f"[ERROR] {csv_path.name}: {error}")

    pd.DataFrame(predictions, columns=["file", "m", "type", "confidence", "branch"]).to_csv(OUT_DIR / "hybrid_test_predictions.csv", index=False)
    print("Phase A complete. Starting Phase B evaluation from GT masks...")

    matches, false_negatives, false_positives, trace_rows = [], [], [], []
    for file_name, pred in cache.items():
        try:
            gt = load_gt(file_name)
            paired, fn, fp = match_events(gt, pred)
            matches.extend({"file": file_name, "gt_m": g["m"], "gt_type": g["type"], "pred_m": p["m"], "pred_type": p["type"], "confidence": p["confidence"], "branch": p["branch"], "distance_m": distance} for g, p, distance in paired)
            false_negatives.extend({"file": file_name, **event} for event in fn)
            false_positives.extend({"file": file_name, **event} for event in fp)
            trace_rows.append({"file": file_name, "TP": len(paired), "FP": len(fp), "FN": len(fn), "gt_events": len(gt), "predicted_events": len(pred)})
        except Exception as error:
            errors.append({"file": file_name, "stage": "evaluation", "error": str(error)})
            print(f"[EVALUATION ERROR] {file_name}: {error}")

    matches_df, fn_df, fp_df = pd.DataFrame(matches), pd.DataFrame(false_negatives), pd.DataFrame(false_positives)
    matches_df.to_csv(OUT_DIR / "hybrid_test_matches.csv", index=False)
    fn_df.to_csv(OUT_DIR / "hybrid_test_false_negatives.csv", index=False)
    fp_df.to_csv(OUT_DIR / "hybrid_test_false_positives.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(OUT_DIR / "hybrid_test_trace_summary.csv", index=False)
    pd.DataFrame(errors).to_csv(OUT_DIR / "hybrid_test_errors.csv", index=False)

    rows = []
    for cls in EVENT_CLASSES:
        tp = int((matches_df["gt_type"] == cls).sum()) if not matches_df.empty else 0
        fp = int((fp_df["type"] == cls).sum()) if not fp_df.empty else 0
        fn = int((fn_df["type"] == cls).sum()) if not fn_df.empty else 0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": cls, "TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1})
    per_class = pd.DataFrame(rows)
    per_class.to_csv(OUT_DIR / "hybrid_test_per_class_metrics.csv", index=False)

    tp, fp, fn = int(per_class.TP.sum()), int(per_class.FP.sum()), int(per_class.FN.sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    overall = pd.DataFrame([{"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1}])
    overall.to_csv(OUT_DIR / "hybrid_test_overall_metrics.csv", index=False)

    print("\nFINAL HYBRID TEST RESULTS")
    print(per_class.round(4).to_string(index=False))
    print(f"\nOverall: TP={tp}, FP={fp}, FN={fn}, precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}")
    print(f"Results saved to: {OUT_DIR}")
