"""
test_4class_event_model_fast.py — ускоренный strict end-to-end тест 4-class OTDR модели.

Ускорения относительно старой версии:
  1. gradient вычисляется 1 раз на CSV, а не для каждого candidate.
  2. Rolling mean/std/slope считаются векторно через cumulative sums.
  3. Нет тысяч np.polyfit() внутри циклов.
  4. Все candidate feature rows для одного CSV передаются модели ОДНИМ batch-вызовом
     predict/predict_proba, а не по одной строке.
  5. peak width ищется только для реально предсказанных event-кандидатов.
  6. Есть прогресс по каждому файлу: candidates / events / elapsed seconds.

Строгость теста:
  - PHASE A prediction читает только CSV; JSON не открывается.
  - PHASE B открывает .mask.json исключительно после окончания prediction всех файлов,
    только чтобы посчитать event-level precision / recall / F1.

Запуск:
  python test_4class_event_model_fast.py --data "C:\\Users\\PCA\\OneDrive - Astana IT University\\Desktop\\OTDR_CLassification\\test_dataset"

Опционально, чтобы быстро проверить на первых 10 файлах:
  python test_4class_event_model_fast.py --data "...\\test_dataset" --max-files 10
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

DEFAULT_MODEL = APP_DIR / "gt_event_model_data_4class" / "model_results" / "best_4class_event_classifier.joblib"
DEFAULT_OUT = APP_DIR / "gt_event_model_data_4class" / "test_results_fast"
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


def rolling_mean_std(x, radius):
    """Centered rolling mean/std in O(n) using cumulative sums."""
    n = len(x)
    left = np.maximum(0, np.arange(n) - radius)
    right = np.minimum(n, np.arange(n) + radius + 1)
    cs = np.r_[0.0, np.cumsum(x, dtype=float)]
    cs2 = np.r_[0.0, np.cumsum(x * x, dtype=float)]
    count = (right - left).astype(float)
    mean = (cs[right] - cs[left]) / count
    var = (cs2[right] - cs2[left]) / count - mean * mean
    return mean, np.sqrt(np.maximum(var, 0.0))


def interval_mean_std(x, left, right):
    """Mean/std for arbitrary [left, right) intervals, vectors of indexes."""
    n = len(x)
    left = np.clip(np.asarray(left, dtype=int), 0, n)
    right = np.clip(np.asarray(right, dtype=int), 0, n)
    count = np.maximum(right - left, 1).astype(float)
    cs = np.r_[0.0, np.cumsum(x, dtype=float)]
    cs2 = np.r_[0.0, np.cumsum(x * x, dtype=float)]
    mean = (cs[right] - cs[left]) / count
    var = (cs2[right] - cs2[left]) / count - mean * mean
    return mean, np.sqrt(np.maximum(var, 0.0))


def local_slopes(km, db, idxs, width):
    """Fast OLS slopes in fixed windows around all candidates at once."""
    slopes_pre = np.zeros(len(idxs), dtype=float)
    slopes_post = np.zeros(len(idxs), dtype=float)
    n = len(db)

    for j, idx in enumerate(idxs):
        a, b = max(0, idx - width), idx
        if b - a >= 3:
            x = km[a:b]
            y = db[a:b]
            xc = x - x.mean()
            denom = np.dot(xc, xc)
            slopes_pre[j] = np.dot(xc, y - y.mean()) / denom if denom > 0 else 0.0

        a, b = idx + 1, min(n, idx + 1 + width)
        if b - a >= 3:
            x = km[a:b]
            y = db[a:b]
            xc = x - x.mean()
            denom = np.dot(xc, xc)
            slopes_post[j] = np.dot(xc, y - y.mean()) / denom if denom > 0 else 0.0

    return slopes_pre, slopes_post


def fast_peak_width(km, db, idx, bg, peak_height, max_search_samples):
    """Bounded peak width: loops only a small local range around one retained event."""
    if peak_height <= 0:
        return 0.0
    half = bg + peak_height / 2.0
    lo = idx
    stop_lo = max(0, idx - max_search_samples)
    while lo > stop_lo and db[lo] > half:
        lo -= 1
    hi = idx
    stop_hi = min(len(db) - 1, idx + max_search_samples)
    while hi < stop_hi and db[hi] > half:
        hi += 1
    return float((km[hi] - km[lo]) * 1000.0)


def candidates(km, db):
    """High-recall candidate generation, vectorized except the small step loop."""
    n = len(db)
    if n < 10:
        return np.array([], dtype=int)

    sm = moving_average(db, m_to_samples(km, 1.0))
    distance = m_to_samples(km, 1.5)
    baseline = np.median(sm)
    mad = np.median(np.abs(sm - baseline)) + 1e-6

    peaks, _ = find_peaks(sm, prominence=max(0.7, 1.5 * mad), distance=distance)

    # Difference of shifted moving medians approximation via local mean for fast candidate proposal.
    # Classifier receives exact raw local features later; this only proposes high-recall locations.
    pre_mean, _ = rolling_mean_std(sm, distance)
    post_mean = np.roll(pre_mean, -distance)
    step = pre_mean - post_mean
    step[-distance:] = 0.0
    threshold = max(0.25, float(np.percentile(step[distance:n-distance], 85))) if n > 2 * distance else 0.25
    down, _ = find_peaks(step, height=threshold, distance=distance)

    mins, _ = find_peaks(-sm, prominence=max(0.25, 0.5 * mad), distance=distance)
    guard = m_to_samples(km, 5.0)
    idxs = np.unique(np.concatenate([peaks, down, mins])).astype(int)
    return idxs[(idxs >= guard) & (idxs < n - guard)]


def build_feature_matrix(km, db, idxs, feature_names, expected_m=None):
    """Builds all 19 feature columns for one trace as one DataFrame."""
    idxs = np.asarray(idxs, dtype=int)
    n = len(db)
    if len(idxs) == 0:
        return pd.DataFrame(columns=feature_names), {}

    ws = m_to_samples(km, 1.5)
    wc = m_to_samples(km, 6.0)
    wsl = m_to_samples(km, 5.0)

    grad = np.gradient(db, km) if n > 2 else np.zeros(n)
    local_mean_all, local_std_all = rolling_mean_std(db, ws)

    left_l = np.maximum(0, idxs - wc)
    left_r = idxs
    right_l = idxs + 1
    right_r = np.minimum(n, idxs + 1 + wc)
    pre_mean, pre_std = interval_mean_std(db, left_l, left_r)
    post_mean, post_std = interval_mean_std(db, right_l, right_r)

    # Local background proxy. Exact median is slower; mean of both contexts is consistent
    # with the other engineered features and much faster for batch inference.
    bg = (pre_mean + post_mean) / 2.0
    local_peak = np.array([np.max(db[max(0, i-ws):min(n, i+ws+1)]) for i in idxs], dtype=float)
    peak_height = local_peak - bg

    pre_slope, post_slope = local_slopes(km, db, idxs, wsl)

    max_pre_deriv = np.array([np.max(grad[max(0, i-ws):i+1]) for i in idxs], dtype=float)
    min_post_deriv = np.array([np.min(grad[i:min(n, i+ws+1)]) for i in idxs], dtype=float)
    local_range = np.array([np.ptp(db[max(0, i-ws):min(n, i+ws+1)]) for i in idxs], dtype=float)

    nominal = float(expected_m) if expected_m and expected_m > 0 else float(km[-1] * 1000.0)
    m_values = km[idxs] * 1000.0

    data = {
        "m_norm": m_values / nominal,
        "db_at_event": db[idxs],
        "local_mean_db": local_mean_all[idxs],
        "local_std_db": local_std_all[idxs],
        "pre_mean_db": pre_mean,
        "post_mean_db": post_mean,
        "loss_dB": pre_mean - post_mean,
        "peak_above_bg_dB": peak_height,
        "pre_slope_dB_per_km": pre_slope,
        "post_slope_dB_per_km": post_slope,
        "slope_change_dB_per_km": post_slope - pre_slope,
        "derivative_at_event_dB_per_km": grad[idxs],
        "max_pre_derivative": max_pre_deriv,
        "min_post_derivative": min_post_deriv,
        "pre_std_db": pre_std,
        "post_std_db": post_std,
        "post_to_pre_std_ratio": post_std / (pre_std + 1e-6),
        "peak_width_m": np.zeros(len(idxs), dtype=float),  # fill only after prediction
        "local_range_db": local_range,
    }

    X = pd.DataFrame(data)
    X = X.reindex(columns=feature_names, fill_value=0.0)
    aux = {"m": m_values, "idxs": idxs, "bg": bg, "peak_height": peak_height, "ws": ws}
    return X, aux


def nms(events, radius_m=5.0):
    """One global NMS, across all types, retaining only the most confident nearby event."""
    kept = []
    for event in sorted(events, key=lambda e: e["confidence"], reverse=True):
        if all(abs(event["m"] - saved["m"]) > radius_m for saved in kept):
            kept.append(event)
    return sorted(kept, key=lambda e: e["m"])


def predict_csv_only_fast(csv_path, bundle, confidence_threshold, expected_m=None):
    """CSV-only prediction. No JSON/mask access in this function."""
    km, db, _ = parse_otdr_csv(csv_path)
    km, db = trim_dead_zone(km, db)
    idxs = candidates(km, db)
    if len(idxs) == 0:
        return [], 0

    model = bundle["model"]
    feature_names = bundle["features"]
    X, aux = build_feature_matrix(km, db, idxs, feature_names, expected_m)

    # ONE BATCH prediction for all candidates in this CSV.
    preds = model.predict(X)
    proba = model.predict_proba(X)
    model_classes = [str(c) for c in model.classes_]

    raw_events = []
    retained_positions = np.where(np.isin(preds.astype(str), list(EVENT_CLASSES)))[0]
    max_width_search = m_to_samples(km, 8.0)

    # peak_width is only computed for non-background events, often a small fraction.
    for pos in retained_positions:
        pred = str(preds[pos])
        cls_pos = model_classes.index(pred)
        confidence = float(proba[pos, cls_pos])
        if confidence < confidence_threshold:
            continue

        idx = int(aux["idxs"][pos])
        width = fast_peak_width(km, db, idx, float(aux["bg"][pos]),
                                float(aux["peak_height"][pos]), max_width_search)
        raw_events.append({
            "m": float(aux["m"][pos]),
            "type": pred,
            "confidence": confidence,
            "candidate_idx": idx,
            "peak_width_m": width,
        })

    return nms(raw_events, radius_m=5.0), len(idxs)


# ------------------- Evaluation only: JSON opens below this marker ---------
def load_gt(mask_path):
    data = json.load(open(mask_path, encoding="utf-8"))
    return [{"m": float(a["m"]), "type": a["type"]}
            for a in data.get("annotations", []) if a.get("type") in EVENT_CLASSES]


def match(gt, pred, tolerance):
    possible = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g["type"] == p["type"]:
                d = abs(g["m"] - p["m"])
                if d <= tolerance:
                    possible.append((d, gi, pi))
    possible.sort()
    used_gt, used_pred, matches = set(), set(), []
    for d, gi, pi in possible:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            matches.append((gt[gi], pred[pi], d))
    fn = [g for i, g in enumerate(gt) if i not in used_gt]
    fp = [p for i, p in enumerate(pred) if i not in used_pred]
    return matches, fn, fp


def main():
    ap = argparse.ArgumentParser(description="Fast strict test for 4-class OTDR event model.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--confidence", type=float, default=0.45)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--expected-m", type=float, default=None)
    ap.add_argument("--max-files", type=int, default=0, help="0=all; otherwise first N files")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model)
    csvs = sorted(glob.glob(os.path.join(args.data, "*.csv")))
    if args.max_files > 0:
        csvs = csvs[:args.max_files]

    print(f"Model: {bundle.get('model_name', 'unknown')} | classes: {bundle.get('classes', [])}")
    print(f"Files: {len(csvs)} | confidence={args.confidence} | tolerance=±{args.tolerance}m")

    all_predictions, all_matches, all_fn, all_fp, errors = [], [], [], [], []
    predictions_by_file = {}
    start_total = time.perf_counter()

    # PHASE A: only CSV is opened here.
    for number, csv_path in enumerate(csvs, 1):
        fname = os.path.basename(csv_path)
        started = time.perf_counter()
        try:
            pred, n_cand = predict_csv_only_fast(csv_path, bundle, args.confidence, args.expected_m)
            predictions_by_file[fname] = pred
            for p in pred:
                all_predictions.append({"file": fname, **p})
            elapsed = time.perf_counter() - started
            print(f"[{number:02d}/{len(csvs)}] {fname} | candidates={n_cand:3d} | events={len(pred):2d} | {elapsed:.2f}s")
        except Exception as e:
            errors.append({"file": fname, "stage": "prediction", "error": str(e)})
            print(f"[prediction error] {fname}: {e}")

    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv(out / "test_predictions_4class_fast.csv", index=False)
    print(f"\nCSV-only phase finished in {time.perf_counter() - start_total:.2f}s. "
          f"Predicted events={len(pred_df)}.")

    # PHASE B: JSON is read only now, strictly for evaluation.
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
                                    "pred_m": p["m"], "pred_type": p["type"],
                                    "confidence": p["confidence"], "distance_m": d})
            for g in fn:
                all_fn.append({"file": fname, **g})
            for p in fp:
                all_fp.append({"file": fname, **p})
        except Exception as e:
            errors.append({"file": fname, "stage": "evaluation", "error": str(e)})

    matches_df = pd.DataFrame(all_matches)
    fn_df = pd.DataFrame(all_fn)
    fp_df = pd.DataFrame(all_fp)
    matches_df.to_csv(out / "test_matches_4class_fast.csv", index=False)
    fn_df.to_csv(out / "test_false_negatives_4class_fast.csv", index=False)
    fp_df.to_csv(out / "test_false_positives_4class_fast.csv", index=False)
    pd.DataFrame(errors).to_csv(out / "test_errors_4class_fast.csv", index=False)

    metric_rows = []
    for cls in ["bend", "connector", "break"]:
        tp = int((matches_df["gt_type"] == cls).sum()) if not matches_df.empty else 0
        fn_count = int((fn_df["type"] == cls).sum()) if not fn_df.empty else 0
        fp_count = int((fp_df["type"] == cls).sum()) if not fp_df.empty else 0
        precision = tp / (tp + fp_count) if tp + fp_count else 0.0
        recall = tp / (tp + fn_count) if tp + fn_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metric_rows.append({"class": cls, "TP": tp, "FP": fp_count, "FN": fn_count,
                            "precision": precision, "recall": recall, "f1": f1})
    per_class = pd.DataFrame(metric_rows)
    per_class.to_csv(out / "test_per_class_metrics_4class_fast.csv", index=False)

    tp = int(per_class["TP"].sum())
    fp = int(per_class["FP"].sum())
    fn = int(per_class["FN"].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    summary = "\n".join([
        "FAST STRICT 4-CLASS END-TO-END TEST",
        "Prediction phase used CSV only. JSON masks were opened only after prediction for evaluation.",
        f"CSV files: {len(csvs)} | predicted files: {len(predictions_by_file)}",
        f"Confidence threshold: {args.confidence} | match tolerance: ±{args.tolerance}m",
        f"Overall TP={tp} FP={fp} FN={fn}",
        f"Overall precision={precision:.4f}",
        f"Overall recall={recall:.4f}",
        f"Overall F1={f1:.4f}",
        "",
        "Per class:",
        per_class.to_string(index=False),
        "",
        f"Errors: {len(errors)}",
        f"Total elapsed: {time.perf_counter() - start_total:.2f}s",
    ])
    (out / "test_metrics_summary_4class_fast.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
