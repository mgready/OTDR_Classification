"""
test_candidate_level_model.py — строгий end-to-end тест candidate-level RF модели.

Модель обучена честно: CSV-only candidates -> JSON labels after candidate generation.
Этот тест использует ТОЧНО ту же candidate generation + feature extraction.

PHASE A (prediction):
  CSV only -> candidates -> features -> RandomForest -> background discarded -> NMS.
  JSON НЕ открывается.

PHASE B (evaluation):
  только после завершения PHASE A читаются .mask.json и считаются TP/FP/FN,
  precision/recall/F1 с type + distance matching.

Run:
  python test_candidate_level_model.py --data "C:\\Users\\PCA\\OneDrive - Astana IT University\\Desktop\\OTDR_CLassification\\test_dataset"

Quick smoke test:
  python test_candidate_level_model.py --data "...\\test_dataset" --max-files 10
"""

import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))
from otdr_common import parse_otdr_csv, trim_dead_zone

DEFAULT_MODEL = APP_DIR / "candidate_event_model_data_balanced" / "model_results" / "best_candidate_level_event_model.joblib"
DEFAULT_OUT = APP_DIR / "candidate_event_model_data_balanced" / "test_results"
EVENT_CLASSES = {"bend", "connector", "break"}


def m_to_samples(km, meters, minimum=2):
    if len(km) < 2:
        return minimum
    dx = float(np.median(np.diff(km)))
    return max(minimum, int((meters / 1000.0) / dx)) if dx > 0 else minimum


def moving_average(x, samples):
    samples = max(3, int(samples) | 1)
    if len(x) < samples:
        return x.astype(float)
    return np.convolve(x, np.ones(samples, dtype=float) / samples, mode="same")


def safe_mean(x, fallback=0.0):
    return float(np.mean(x)) if len(x) else float(fallback)


def safe_std(x):
    return float(np.std(x)) if len(x) > 1 else 0.0


def local_slope(km_seg, db_seg):
    if len(km_seg) < 3 or np.ptp(km_seg) <= 0:
        return 0.0
    return float(np.polyfit(km_seg, db_seg, 1)[0])


def peak_width(km, db, idx, bg, peak_height):
    if peak_height <= 0:
        return 0.0
    half = bg + peak_height / 2.0
    lo = idx
    while lo > 0 and db[lo] > half:
        lo -= 1
    hi = idx
    while hi < len(db) - 1 and db[hi] > half:
        hi += 1
    return float((km[hi] - km[lo]) * 1000.0)


def candidate_indices_from_csv(km, db):
    """IDENTICAL candidate generator to build_candidate_train_val.py. CSV only."""
    n = len(db)
    if n < 10:
        return []

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
    idxs = np.unique(np.concatenate([peaks, down, mins])).astype(int)
    return sorted(i for i in idxs if guard <= i < n - guard)


def extract_features(km, db, idx, expected_length_m=None):
    """IDENTICAL features to build_candidate_train_val.py."""
    n = len(db)
    ws = m_to_samples(km, 1.5)
    wc = m_to_samples(km, 6.0)
    wsl = m_to_samples(km, 5.0)

    left = db[max(0, idx-wc):idx]
    right = db[idx+1:min(n, idx+1+wc)]
    local = db[max(0, idx-ws):min(n, idx+ws+1)]

    pre = safe_mean(left, db[idx])
    post = safe_mean(right, db[idx])
    bg_values = np.concatenate([left, right]) if len(left) + len(right) else np.array([db[idx]])
    bg = float(np.median(bg_values))
    local_peak = float(np.max(local)) if len(local) else float(db[idx])
    peak_height = local_peak - bg

    pre_slope = local_slope(km[max(0, idx-wsl):idx], db[max(0, idx-wsl):idx])
    post_slope = local_slope(km[idx+1:min(n, idx+1+wsl)], db[idx+1:min(n, idx+1+wsl)])
    grad = np.gradient(db, km) if n > 2 else np.zeros(n)
    pre_grad = grad[max(0, idx-ws):idx+1]
    post_grad = grad[idx:min(n, idx+ws+1)]

    nominal = float(expected_length_m) if expected_length_m and expected_length_m > 0 else float(km[-1] * 1000.0)
    return {
        "m": float(km[idx] * 1000.0),
        "candidate_idx": int(idx),
        "m_norm": float(km[idx] * 1000.0 / nominal),
        "db_at_event": float(db[idx]),
        "local_mean_db": safe_mean(local, db[idx]),
        "local_std_db": safe_std(local),
        "pre_mean_db": pre,
        "post_mean_db": post,
        "loss_dB": float(pre-post),
        "peak_above_bg_dB": float(peak_height),
        "pre_slope_dB_per_km": pre_slope,
        "post_slope_dB_per_km": post_slope,
        "slope_change_dB_per_km": post_slope-pre_slope,
        "derivative_at_event_dB_per_km": float(grad[idx]),
        "max_pre_derivative": float(np.max(pre_grad)) if len(pre_grad) else 0.0,
        "min_post_derivative": float(np.min(post_grad)) if len(post_grad) else 0.0,
        "pre_std_db": safe_std(left),
        "post_std_db": safe_std(right),
        "post_to_pre_std_ratio": float(safe_std(right)/(safe_std(left)+1e-6)),
        "peak_width_m": peak_width(km, db, idx, bg, peak_height),
        "local_range_db": float(np.ptp(local)) if len(local) else 0.0,
    }


def nms_global(events, radius_m=5.0):
    """One physical event in a radius gets one highest-confidence predicted class."""
    kept = []
    for e in sorted(events, key=lambda x: x["confidence"], reverse=True):
        if all(abs(e["m"] - old["m"]) > radius_m for old in kept):
            kept.append(e)
    return sorted(kept, key=lambda x: x["m"])


def predict_from_csv_only(csv_path, bundle, threshold, expected_m=None):
    """STRICT: no JSON access in this function."""
    km, db, _ = parse_otdr_csv(csv_path)
    km, db = trim_dead_zone(km, db)
    idxs = candidate_indices_from_csv(km, db)
    if not idxs:
        return [], 0

    # Build whole batch then RF inference once per trace.
    feature_rows = [extract_features(km, db, idx, expected_m) for idx in idxs]
    features = bundle["features"]
    X = pd.DataFrame([{f: r[f] for f in features} for r in feature_rows], columns=features)
    model = bundle["model"]
    preds = model.predict(X).astype(str)
    probas = model.predict_proba(X)
    model_classes = [str(c) for c in model.classes_]

    events = []
    for row, pred, proba in zip(feature_rows, preds, probas):
        if pred not in EVENT_CLASSES:
            continue
        confidence = float(proba[model_classes.index(pred)])
        if confidence >= threshold:
            events.append({"m": row["m"], "type": pred, "confidence": confidence,
                           "candidate_idx": row["candidate_idx"], "loss_dB": row["loss_dB"],
                           "peak_above_bg_dB": row["peak_above_bg_dB"]})
    return nms_global(events, radius_m=5.0), len(idxs)


# JSON is accessed only below this line, strictly for scoring after prediction.
def load_gt(mask_path):
    data = json.load(open(mask_path, encoding="utf-8"))
    return [{"m": float(a["m"]), "type": a["type"]}
            for a in data.get("annotations", []) if a.get("type") in EVENT_CLASSES]


def match_events(gt, pred, tolerance_m):
    possible = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g["type"] == p["type"]:
                d = abs(g["m"] - p["m"])
                if d <= tolerance_m:
                    possible.append((d, gi, pi))
    possible.sort()

    used_gt, used_pred, matched = set(), set(), []
    for d, gi, pi in possible:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi); used_pred.add(pi)
            matched.append((gt[gi], pred[pi], d))
    fn = [g for i, g in enumerate(gt) if i not in used_gt]
    fp = [p for i, p in enumerate(pred) if i not in used_pred]
    return matched, fn, fp


def main():
    ap = argparse.ArgumentParser(description="Strict candidate-level RF test: CSV prediction then JSON evaluation.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--threshold", type=float, default=None,
                    help="None means recommended confidence threshold saved inside joblib")
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--expected-m", type=float, default=None,
                    help="optional known cable length; never read from test JSON")
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    threshold = args.threshold if args.threshold is not None else float(bundle.get("recommended_confidence_threshold", 0.50))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    csvs = sorted(glob.glob(os.path.join(args.data, "*.csv")))
    if args.max_files > 0:
        csvs = csvs[:args.max_files]

    print(f"Model: {bundle.get('model_name', 'unknown')}")
    print(f"Model selection metric: {bundle.get('selection_metric', 'unknown')}")
    print(f"Confidence threshold: {threshold:.2f} | Match tolerance: ±{args.tolerance:.1f}m")
    print(f"Test CSV files: {len(csvs)}")

    predictions_by_file = {}
    prediction_rows, matches_rows, fn_rows, fp_rows, error_rows = [], [], [], [], []
    t0 = time.perf_counter()

    # ============== PHASE A: CSV ONLY PREDICTION ==============
    for no, csv_path in enumerate(csvs, 1):
        fname = os.path.basename(csv_path)
        t_file = time.perf_counter()
        try:
            events, n_candidates = predict_from_csv_only(csv_path, bundle, threshold, args.expected_m)
            predictions_by_file[fname] = events
            for e in events:
                prediction_rows.append({"file": fname, "pred_m": e["m"], "pred_type": e["type"],
                                        "confidence": e["confidence"], "candidate_idx": e["candidate_idx"],
                                        "loss_dB": e["loss_dB"], "peak_above_bg_dB": e["peak_above_bg_dB"]})
            print(f"[{no:02d}/{len(csvs)}] {fname} | candidates={n_candidates:3d} | events={len(events):2d} | {time.perf_counter()-t_file:.2f}s")
        except Exception as e:
            error_rows.append({"file": fname, "stage": "prediction", "error": str(e)})
            print(f"[prediction error] {fname}: {e}")

    prediction_df = pd.DataFrame(prediction_rows)
    prediction_df.to_csv(out / "test_predictions.csv", index=False)
    print(f"\nCSV-only prediction complete in {time.perf_counter()-t0:.2f}s; events={len(prediction_df)}")

    # ============== PHASE B: JSON ONLY FOR EVALUATION ==============
    n_evaluated = 0
    for csv_path in csvs:
        fname = os.path.basename(csv_path)
        mask_path = os.path.splitext(csv_path)[0] + ".mask.json"
        if fname not in predictions_by_file or not os.path.exists(mask_path):
            continue
        try:
            gt = load_gt(mask_path)
            pred = predictions_by_file[fname]
            matched, fn, fp = match_events(gt, pred, args.tolerance)
            n_evaluated += 1
            for g, p, d in matched:
                matches_rows.append({"file": fname, "gt_m": g["m"], "gt_type": g["type"],
                                     "pred_m": p["m"], "pred_type": p["type"],
                                     "confidence": p["confidence"], "distance_m": d})
            for g in fn:
                fn_rows.append({"file": fname, "gt_m": g["m"], "gt_type": g["type"]})
            for p in fp:
                fp_rows.append({"file": fname, "pred_m": p["m"], "pred_type": p["type"],
                                "confidence": p["confidence"]})
        except Exception as e:
            error_rows.append({"file": fname, "stage": "evaluation", "error": str(e)})

    matches_df = pd.DataFrame(matches_rows)
    fn_df = pd.DataFrame(fn_rows)
    fp_df = pd.DataFrame(fp_rows)
    matches_df.to_csv(out / "test_matches.csv", index=False)
    fn_df.to_csv(out / "test_false_negatives.csv", index=False)
    fp_df.to_csv(out / "test_false_positives.csv", index=False)
    pd.DataFrame(error_rows).to_csv(out / "test_errors.csv", index=False)

    metric_rows = []
    for cls in ["bend", "connector", "break"]:
        tp = int((matches_df["gt_type"] == cls).sum()) if not matches_df.empty else 0
        fp = int((fp_df["pred_type"] == cls).sum()) if not fp_df.empty else 0
        fn = int((fn_df["gt_type"] == cls).sum()) if not fn_df.empty else 0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metric_rows.append({"class": cls, "TP": tp, "FP": fp, "FN": fn,
                            "precision": precision, "recall": recall, "f1": f1})
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out / "test_per_class_metrics.csv", index=False)

    tp = int(metrics_df["TP"].sum()); fp = int(metrics_df["FP"].sum()); fn = int(metrics_df["FN"].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    summary = "\n".join([
        "STRICT CANDIDATE-LEVEL END-TO-END TEST",
        "Phase A used CSV only. JSON masks were opened only after all predictions, for evaluation.",
        f"CSV files: {len(csvs)} | evaluated masks: {n_evaluated}",
        f"Model: {bundle.get('model_name', 'unknown')}",
        f"Confidence threshold: {threshold:.2f} | Matching tolerance: ±{args.tolerance:.1f}m",
        f"Overall: TP={tp}, FP={fp}, FN={fn}",
        f"Overall precision={precision:.4f}",
        f"Overall recall={recall:.4f}",
        f"Overall F1={f1:.4f}",
        "", "Per class:", metrics_df.to_string(index=False),
        "", f"Errors: {len(error_rows)}", f"Elapsed seconds: {time.perf_counter()-t0:.2f}",
    ])
    (out / "test_metrics_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nSaved results: {out}")


if __name__ == "__main__":
    main()
