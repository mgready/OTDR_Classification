"""
test_4class_event_model.py — end-to-end test 4-class ExtraTrees model.

Prediction phase:
  - uses CSV only;
  - generates candidate positions;
  - extracts the same 19 features as training;
  - predicts bend/connector/break/background;
  - discards background.

Evaluation phase:
  - starts only after all CSV-only predictions are complete;
  - then reads .mask.json and calculates event-level metrics.

Default model:
  gt_event_model_data_4class/model_results/best_4class_event_classifier.joblib

Run:
  python test_4class_event_model.py --data "C:\\...\\test_dataset"
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))
from otdr_common import parse_otdr_csv, trim_dead_zone

DEFAULT_MODEL = APP_DIR / "gt_event_model_data_4class" / "model_results" / "best_4class_event_classifier.joblib"
DEFAULT_OUT = APP_DIR / "gt_event_model_data_4class" / "test_results"
FEATURE_COLS = [
    "m_norm", "db_at_event", "local_mean_db", "local_std_db",
    "pre_mean_db", "post_mean_db", "loss_dB", "peak_above_bg_dB",
    "pre_slope_dB_per_km", "post_slope_dB_per_km", "slope_change_dB_per_km",
    "derivative_at_event_dB_per_km", "max_pre_derivative", "min_post_derivative",
    "pre_std_db", "post_std_db", "post_to_pre_std_ratio", "peak_width_m",
    "local_range_db",
]
EVENT_CLASSES = {"bend", "connector", "break"}


def m_to_samples(km, meters, minimum=2):
    if len(km) < 2:
        return minimum
    dx = float(np.median(np.diff(km)))
    return max(minimum, int((meters / 1000.0) / dx)) if dx > 0 else minimum


def safe_mean(x, fallback=0.0):
    return float(np.mean(x)) if len(x) else float(fallback)


def safe_std(x):
    return float(np.std(x)) if len(x) > 1 else 0.0


def local_slope(x, y):
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 3 and np.ptp(x) > 0 else 0.0


def peak_width(km, db, idx, bg, height):
    if height <= 0:
        return 0.0
    half = bg + height / 2.0
    lo = idx
    while lo > 0 and db[lo] > half:
        lo -= 1
    hi = idx
    while hi < len(db) - 1 and db[hi] > half:
        hi += 1
    return float((km[hi] - km[lo]) * 1000.0)


def extract_features(km, db, idx, expected_m=None):
    n = len(db)
    ws = m_to_samples(km, 1.5)
    wc = m_to_samples(km, 6.0)
    wsl = m_to_samples(km, 5.0)

    left = db[max(0, idx - wc):idx]
    right = db[idx + 1:min(n, idx + 1 + wc)]
    local = db[max(0, idx - ws):min(n, idx + ws + 1)]

    pre = safe_mean(left, db[idx])
    post = safe_mean(right, db[idx])
    bg_values = np.concatenate([left, right]) if len(left) + len(right) else np.array([db[idx]])
    bg = float(np.median(bg_values))
    peak = float(np.max(local)) if len(local) else float(db[idx])
    peak_height = peak - bg

    pre_slope = local_slope(km[max(0, idx - wsl):idx], db[max(0, idx - wsl):idx])
    post_slope = local_slope(km[idx + 1:min(n, idx + 1 + wsl)], db[idx + 1:min(n, idx + 1 + wsl)])
    grad = np.gradient(db, km) if n > 2 else np.zeros(n)
    pre_grad = grad[max(0, idx - ws):idx + 1]
    post_grad = grad[idx:min(n, idx + ws + 1)]

    nominal = expected_m if expected_m and expected_m > 0 else float(km[-1] * 1000.0)
    row = {
        "m_norm": float(km[idx] * 1000.0 / nominal),
        "db_at_event": float(db[idx]),
        "local_mean_db": safe_mean(local, db[idx]),
        "local_std_db": safe_std(local),
        "pre_mean_db": pre,
        "post_mean_db": post,
        "loss_dB": float(pre - post),
        "peak_above_bg_dB": float(peak_height),
        "pre_slope_dB_per_km": pre_slope,
        "post_slope_dB_per_km": post_slope,
        "slope_change_dB_per_km": post_slope - pre_slope,
        "derivative_at_event_dB_per_km": float(grad[idx]),
        "max_pre_derivative": float(np.max(pre_grad)) if len(pre_grad) else 0.0,
        "min_post_derivative": float(np.min(post_grad)) if len(post_grad) else 0.0,
        "pre_std_db": safe_std(left),
        "post_std_db": safe_std(right),
        "post_to_pre_std_ratio": float(safe_std(right) / (safe_std(left) + 1e-6)),
        "peak_width_m": peak_width(km, db, idx, bg, peak_height),
        "local_range_db": float(np.ptp(local)) if len(local) else 0.0,
    }
    return row


def moving_average(x, samples):
    samples = max(3, int(samples) | 1)
    return np.convolve(x, np.ones(samples) / samples, mode="same") if len(x) >= samples else x.astype(float)


def candidates(km, db):
    n = len(db)
    sm = moving_average(db, m_to_samples(km, 1.0))
    distance = m_to_samples(km, 1.5)
    baseline = np.median(sm)
    mad = np.median(np.abs(sm - baseline)) + 1e-6

    peaks, _ = find_peaks(sm, prominence=max(0.7, 1.5 * mad), distance=distance)
    step = np.zeros(n)
    for i in range(distance, n - distance):
        step[i] = np.median(sm[i-distance:i]) - np.median(sm[i:i+distance])
    threshold = max(0.25, np.percentile(step[distance:n-distance], 85)) if n > 2 * distance else 0.25
    down, _ = find_peaks(step, height=threshold, distance=distance)
    mins, _ = find_peaks(-sm, prominence=max(0.25, 0.5 * mad), distance=distance)

    guard = m_to_samples(km, 5.0)
    return sorted(set(int(i) for i in np.concatenate([peaks, down, mins])
                      if guard <= i < n - guard))


def nms(events, radius_m=5.0):
    """Suppress nearby predictions. Uses confidence across ALL event types,
    preventing bend and connector duplicates at the same physical location."""
    result = []
    for event in sorted(events, key=lambda x: x["confidence"], reverse=True):
        if all(abs(event["m"] - kept["m"]) > radius_m for kept in result):
            result.append(event)
    return sorted(result, key=lambda x: x["m"])


def predict_csv_only(csv_path, bundle, confidence_threshold, expected_m=None):
    """No JSON access here."""
    km, db, _ = parse_otdr_csv(csv_path)
    km, db = trim_dead_zone(km, db)
    model = bundle["model"]
    features = bundle["features"]
    classes = list(model.classes_) if hasattr(model, "classes_") else list(bundle["classes"])
    events = []

    for idx in candidates(km, db):
        row = extract_features(km, db, idx, expected_m)
        X = pd.DataFrame([[row[c] for c in features]], columns=features)
        pred = str(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        pmap = {str(c): float(p) for c, p in zip(classes, proba)}
        confidence = pmap[pred]
        if pred in EVENT_CLASSES and confidence >= confidence_threshold:
            events.append({
                "m": float(km[idx] * 1000.0), "type": pred,
                "confidence": float(confidence), "candidate_idx": int(idx),
            })
    return nms(events)


def load_gt(mask_path):
    data = json.load(open(mask_path, encoding="utf-8"))
    return [{"m": float(a["m"]), "type": a["type"]}
            for a in data.get("annotations", []) if a.get("type") in EVENT_CLASSES]


def match(gt, pred, tolerance):
    pairs = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g["type"] == p["type"] and abs(g["m"] - p["m"]) <= tolerance:
                pairs.append((abs(g["m"] - p["m"]), gi, pi))
    pairs.sort()
    used_g, used_p, matches = set(), set(), []
    for d, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi); used_p.add(pi); matches.append((gt[gi], pred[pi], d))
    fn = [g for i, g in enumerate(gt) if i not in used_g]
    fp = [p for i, p in enumerate(pred) if i not in used_p]
    return matches, fn, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--confidence", type=float, default=0.45)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--expected-m", type=float, default=None,
                    help="optional nominal length; not taken from test JSON")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model)
    csvs = sorted(glob.glob(os.path.join(args.data, "*.csv")))

    all_predictions, all_matches, all_fn, all_fp, errors = [], [], [], [], []
    predictions_by_file = {}

    # PHASE A: CSV-only predictions.
    for csv_path in csvs:
        fname = os.path.basename(csv_path)
        try:
            pred = predict_csv_only(csv_path, bundle, args.confidence, args.expected_m)
            predictions_by_file[fname] = pred
            for p in pred:
                all_predictions.append({"file": fname, **p})
        except Exception as e:
            errors.append({"file": fname, "stage": "prediction", "error": str(e)})
            print(f"[prediction error] {fname}: {e}")

    pd.DataFrame(all_predictions).to_csv(out / "test_predictions_4class.csv", index=False)
    print(f"CSV-only prediction complete: {len(all_predictions)} events across {len(predictions_by_file)} files.")

    # PHASE B: only now read JSON for evaluation.
    for csv_path in csvs:
        fname = os.path.basename(csv_path)
        mask_path = os.path.splitext(csv_path)[0] + ".mask.json"
        if fname not in predictions_by_file or not os.path.exists(mask_path):
            continue
        try:
            gt = load_gt(mask_path)
            pred = predictions_by_file[fname]
            matches, fn, fp = match(gt, pred, args.tolerance)
            for g, p, d in matches:
                all_matches.append({"file": fname, "gt_m": g["m"], "gt_type": g["type"],
                                    "pred_m": p["m"], "pred_type": p["type"], "distance_m": d})
            for g in fn:
                all_fn.append({"file": fname, **g})
            for p in fp:
                all_fp.append({"file": fname, **p})
        except Exception as e:
            errors.append({"file": fname, "stage": "evaluation", "error": str(e)})

    matches_df = pd.DataFrame(all_matches)
    fn_df = pd.DataFrame(all_fn)
    fp_df = pd.DataFrame(all_fp)
    matches_df.to_csv(out / "test_matches_4class.csv", index=False)
    fn_df.to_csv(out / "test_false_negatives_4class.csv", index=False)
    fp_df.to_csv(out / "test_false_positives_4class.csv", index=False)
    pd.DataFrame(errors).to_csv(out / "test_errors_4class.csv", index=False)

    rows = []
    for cls in ["bend", "connector", "break"]:
        tp = int((matches_df["gt_type"] == cls).sum()) if not matches_df.empty else 0
        fn = int((fn_df["type"] == cls).sum()) if not fn_df.empty else 0
        fp = int((fp_df["type"] == cls).sum()) if not fp_df.empty else 0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": cls, "TP": tp, "FP": fp, "FN": fn,
                     "precision": precision, "recall": recall, "f1": f1})
    per_class = pd.DataFrame(rows)
    per_class.to_csv(out / "test_per_class_metrics_4class.csv", index=False)

    tp = int(per_class["TP"].sum()); fp = int(per_class["FP"].sum()); fn = int(per_class["FN"].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    summary = "\n".join([
        "STRICT 4-CLASS END-TO-END TEST",
        "Prediction used CSV only; masks were opened only after predictions for evaluation.",
        f"CSV files: {len(csvs)} | prediction files: {len(predictions_by_file)}",
        f"Confidence threshold: {args.confidence} | matching tolerance: ±{args.tolerance} m",
        f"Overall TP={tp} FP={fp} FN={fn}",
        f"Overall precision={precision:.4f}",
        f"Overall recall={recall:.4f}",
        f"Overall F1={f1:.4f}",
        "",
        "Per class:",
        per_class.to_string(index=False),
        "",
        f"Errors: {len(errors)}",
    ])
    (out / "test_metrics_summary_4class.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nSaved results: {out}")


if __name__ == "__main__":
    main()
