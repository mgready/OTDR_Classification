"""
test_gt_event_model.py — ЧЕСТНАЯ event-level оценка best_gt_event_classifier.joblib
на test_dataset.

КРИТИЧЕСКОЕ ПРАВИЛО:
  - При ПРЕДСКАЗАНИИ этот скрипт НЕ читает никакие JSON-маски.
  - Модель получает только CSV рефлектограмму.
  - JSON открывается ТОЛЬКО ПОСЛЕ получения всех предсказаний — для сравнения
    prediction vs ground truth и расчёта precision/recall/F1.

Пайплайн на каждом test CSV:
  A. prediction_from_csv_only(csv):
     - ищет CANDIDATE positions на всей трассе по локальным extrema/изменениям;
     - вычисляет ровно те же 19 local features, на которых обучалась модель;
     - классифицирует кандидаты как bend/connector/break;
     - применяет NMS (non-maximum suppression), чтобы несколько близких
       кандидатов не стали несколькими одинаковыми событиями.
  B. evaluate_after_prediction(...):
     - только теперь открывает .mask.json;
     - сравнивает predicted events с annotations, если маска существует;
     - match: тот же type + позиция в пределах tolerance_m.

ВНИМАНИЕ: train notebook обучал модель на точных GT-позициях. Для полного
end-to-end теста здесь нужна candidate generation. Поэтому метрики этого
скрипта будут более строгими и, вероятно, ниже, чем 0.9899 F1 validation:
они проверяют одновременно (1) найти позицию и (2) правильно назвать тип.

Запуск:
    python test_gt_event_model.py --data "C:\\Users\\PCA\\OneDrive - Astana IT University\\Desktop\\OTDR_CLassification\\test_dataset"

Результаты:
  gt_event_model_data/test_results/
    - test_predictions.csv      : предсказания по каждому CSV (до GT comparison)
    - test_matches.csv          : совпавшие события
    - test_false_negatives.csv  : GT-события, которые модель пропустила
    - test_false_positives.csv  : предсказания без соответствия в GT
    - test_metrics_summary.txt  : итоговые Precision/Recall/F1
    - test_per_class_metrics.csv
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Код предполагается положить рядом с otdr_common.py и train notebook output.
APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))

from otdr_common import parse_otdr_csv, trim_dead_zone

DEFAULT_MODEL = APP_DIR / "gt_event_model_data" / "model_results" / "best_gt_event_classifier.joblib"
DEFAULT_OUT = APP_DIR / "gt_event_model_data" / "test_results"


def m_to_samples(km, meters, minimum=2):
    if len(km) < 2:
        return minimum
    dx = float(np.median(np.diff(km)))
    if dx <= 0:
        return minimum
    return max(minimum, int((meters / 1000.0) / dx))


def safe_mean(x, fallback=0.0):
    return float(np.mean(x)) if len(x) else float(fallback)


def safe_std(x):
    return float(np.std(x)) if len(x) > 1 else 0.0


def local_slope(km_seg, db_seg):
    if len(km_seg) < 3 or np.ptp(km_seg) <= 0:
        return 0.0
    return float(np.polyfit(km_seg, db_seg, 1)[0])


def peak_width(km, db, idx, bg, peak_above_bg):
    if peak_above_bg <= 0:
        return 0.0
    half = bg + peak_above_bg / 2.0
    lo = idx
    while lo > 0 and db[lo] > half:
        lo -= 1
    hi = idx
    while hi < len(db) - 1 and db[hi] > half:
        hi += 1
    return float((km[hi] - km[lo]) * 1000.0)


def extract_features(km, db, idx, expected_length_m=None):
    """Идентичный feature extractor тому, что использовался в split_gt_events.py,
    кроме входа: там annotation_m -> idx, здесь idx уже получен candidate detector'ом."""
    n = len(db)
    w_small = m_to_samples(km, 1.5)
    w_context = m_to_samples(km, 6.0)
    w_slope = m_to_samples(km, 5.0)

    left_ctx = db[max(0, idx - w_context):idx]
    right_ctx = db[idx + 1:min(n, idx + 1 + w_context)]
    local = db[max(0, idx - w_small):min(n, idx + w_small + 1)]

    pre_mean = safe_mean(left_ctx, db[idx])
    post_mean = safe_mean(right_ctx, db[idx])
    local_mean = safe_mean(local, db[idx])
    pre_std = safe_std(left_ctx)
    post_std = safe_std(right_ctx)
    local_std = safe_std(local)
    loss = pre_mean - post_mean

    bg_values = np.concatenate([left_ctx, right_ctx]) if len(left_ctx) + len(right_ctx) else np.array([db[idx]])
    bg = float(np.median(bg_values))
    local_peak = float(np.max(local)) if len(local) else float(db[idx])
    peak_above_bg = local_peak - bg

    pre_i0 = max(0, idx - w_slope)
    post_i1 = min(n, idx + 1 + w_slope)
    pre_slope = local_slope(km[pre_i0:idx], db[pre_i0:idx])
    post_slope = local_slope(km[idx + 1:post_i1], db[idx + 1:post_i1])

    grad = np.gradient(db, km) if n > 2 else np.zeros(n)
    pre_grad = grad[max(0, idx - w_small):idx + 1]
    post_grad = grad[idx:min(n, idx + 1 + w_small)]

    trace_length_m = float(km[-1] * 1000.0)
    nominal_m = float(expected_length_m) if expected_length_m and expected_length_m > 0 else trace_length_m

    return {
        "m": float(km[idx] * 1000.0),
        "m_norm": float((km[idx] * 1000.0) / nominal_m),
        "db_at_event": float(db[idx]),
        "local_mean_db": float(local_mean),
        "local_std_db": float(local_std),
        "pre_mean_db": float(pre_mean),
        "post_mean_db": float(post_mean),
        "loss_dB": float(loss),
        "peak_above_bg_dB": float(peak_above_bg),
        "pre_slope_dB_per_km": float(pre_slope),
        "post_slope_dB_per_km": float(post_slope),
        "slope_change_dB_per_km": float(post_slope - pre_slope),
        "derivative_at_event_dB_per_km": float(grad[idx]),
        "max_pre_derivative": float(np.max(pre_grad)) if len(pre_grad) else 0.0,
        "min_post_derivative": float(np.min(post_grad)) if len(post_grad) else 0.0,
        "pre_std_db": float(pre_std),
        "post_std_db": float(post_std),
        "post_to_pre_std_ratio": float(post_std / (pre_std + 1e-6)),
        "peak_width_m": float(peak_width(km, db, idx, bg, peak_above_bg)),
        "local_range_db": float(np.ptp(local)) if len(local) else 0.0,
    }


def moving_average(x, samples):
    samples = max(3, int(samples) | 1)
    if len(x) < samples:
        return x.astype(float)
    return np.convolve(x, np.ones(samples) / samples, mode="same")


def candidate_indices(km, db):
    """Генерирует кандидаты ПО CSV, без JSON.

    Используются три источника: сильные positive peaks (connector), сильные
    локальные downward changes (bend/break) и локальные minima после спадов.
    Объединение + minimum distance по метрам сохраняет high-recall; модель
    позже решает тип кандидата.
    """
    n = len(db)
    if n < 10:
        return []

    w_smooth = m_to_samples(km, 1.0)
    sm = moving_average(db, w_smooth)
    dist = m_to_samples(km, 1.5)

    # Robust scale сигналов для порогов, без ручной привязки к абсолютному dB.
    baseline = np.median(sm)
    mad = np.median(np.abs(sm - baseline)) + 1e-6

    # Отражающие высокие пики; prominence малый специально ради recall.
    peaks, _ = find_peaks(sm, prominence=max(0.7, 1.5 * mad), distance=dist)

    # Переход/потеря: разность медиан до/после точки.
    step = np.zeros(n)
    for i in range(dist, n - dist):
        step[i] = np.median(sm[i - dist:i]) - np.median(sm[i:i + dist])
    step_thr = max(0.25, np.percentile(step[dist:n-dist], 85)) if n > 2 * dist else 0.25
    down, _ = find_peaks(step, height=step_thr, distance=dist)

    # Локальные минимумы полезны для точки break после отражающего пика/спада.
    mins, _ = find_peaks(-sm, prominence=max(0.25, 0.5 * mad), distance=dist)

    # Убираем самые первые 5м: launch/dead zone не является кабельным событием.
    guard = m_to_samples(km, 5.0)
    candidates = sorted(set(int(i) for i in np.concatenate([peaks, down, mins])
                            if guard <= i < n - guard))
    return candidates


def nms_by_type(events, radius_m=5.0):
    """Оставляет максимум один наиболее уверенный event каждого типа в радиусе radius_m."""
    kept = []
    for event_type in sorted({e["pred_type"] for e in events}):
        group = sorted([e for e in events if e["pred_type"] == event_type],
                       key=lambda e: e["confidence"], reverse=True)
        chosen = []
        for e in group:
            if all(abs(e["pred_m"] - c["pred_m"]) > radius_m for c in chosen):
                chosen.append(e)
        kept.extend(chosen)
    return sorted(kept, key=lambda e: e["pred_m"])


def prediction_from_csv_only(csv_path, bundle, expected_length_m=None, confidence_threshold=0.45):
    """ПРЕДСКАЗАНИЕ ТОЛЬКО ИЗ CSV. JSON здесь принципиально не открывается."""
    km, db, _ = parse_otdr_csv(csv_path)
    km, db = trim_dead_zone(km, db)
    idxs = candidate_indices(km, db)

    features = bundle["features"]
    model = bundle["model"]
    classes = list(model.classes_) if hasattr(model, "classes_") else list(bundle["classes"])

    raw_events = []
    for idx in idxs:
        feat = extract_features(km, db, idx, expected_length_m)
        X = pd.DataFrame([[feat[c] for c in features]], columns=features)
        pred = str(model.predict(X)[0])
        probas = model.predict_proba(X)[0]
        pmap = {str(c): float(p) for c, p in zip(classes, probas)}
        conf = pmap[pred]

        if conf >= confidence_threshold:
            raw_events.append({
                "pred_m": round(feat["m"], 3),
                "pred_type": pred,
                "confidence": round(conf, 5),
                "loss_dB": round(feat["loss_dB"], 4),
                "peak_above_bg_dB": round(feat["peak_above_bg_dB"], 4),
                "candidate_idx": int(idx),
            })

    return nms_by_type(raw_events, radius_m=5.0)


# ---------------------------------------------------------------------------
# Everything below this line is evaluation-only: JSON is used ONLY after
# prediction_from_csv_only() has returned all predictions for a file.
# ---------------------------------------------------------------------------
def load_ground_truth(mask_path):
    data = json.load(open(mask_path, encoding="utf-8"))
    keep = {"bend", "connector", "break"}
    return [{"gt_m": float(a["m"]), "gt_type": a["type"]}
            for a in data.get("annotations", []) if a.get("type") in keep]


def match_events(gt_events, pred_events, tolerance_m):
    """One-to-one matching requiring BOTH position and type equality."""
    candidates = []
    for gi, g in enumerate(gt_events):
        for pi, p in enumerate(pred_events):
            if g["gt_type"] == p["pred_type"]:
                d = abs(g["gt_m"] - p["pred_m"])
                if d <= tolerance_m:
                    candidates.append((d, gi, pi))
    candidates.sort()

    used_gt, used_pred = set(), set()
    matches = []
    for d, gi, pi in candidates:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            matches.append((gt_events[gi], pred_events[pi], d))

    fn = [g for gi, g in enumerate(gt_events) if gi not in used_gt]
    fp = [p for pi, p in enumerate(pred_events) if pi not in used_pred]
    return matches, fn, fp


def main():
    ap = argparse.ArgumentParser(description="Strict CSV-only prediction followed by JSON-only evaluation.")
    ap.add_argument("--data", required=True, help="test_dataset folder")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--confidence", type=float, default=0.45)
    ap.add_argument("--expected-m", type=float, default=None,
                    help="optional known nominal fibre length; JSON is not used here")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model)
    print(f"Model: {bundle.get('model_name', 'unknown')} | classes: {bundle.get('classes', [])}")
    print(f"Prediction confidence threshold: {args.confidence}; matching tolerance: ±{args.tolerance} m")

    csv_paths = sorted(glob.glob(os.path.join(args.data, "*.csv")))
    prediction_rows, matches_rows, fn_rows, fp_rows = [], [], [], []
    errors = []

    # PHASE A: produce all predictions from CSV only.
    predictions_by_file = {}
    for csv_path in csv_paths:
        fname = os.path.basename(csv_path)
        try:
            pred_events = prediction_from_csv_only(
                csv_path, bundle, expected_length_m=args.expected_m,
                confidence_threshold=args.confidence,
            )
            predictions_by_file[fname] = pred_events
            for p in pred_events:
                prediction_rows.append({"file": fname, **p})
        except Exception as e:
            errors.append({"file": fname, "stage": "prediction", "error": str(e)})
            print(f"[prediction error] {fname}: {e}")

    # Save predictions BEFORE loading any JSON. This is an audit trail proving prediction was CSV-only.
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"CSV-only prediction complete: {len(pred_df)} predicted events across {len(predictions_by_file)} files.")

    # PHASE B: Evaluation only. JSON is opened here and never passed to model/predictor.
    n_eval_files = 0
    for csv_path in csv_paths:
        fname = os.path.basename(csv_path)
        mask_path = os.path.splitext(csv_path)[0] + ".mask.json"
        if not os.path.exists(mask_path) or fname not in predictions_by_file:
            continue
        try:
            gt_events = load_ground_truth(mask_path)
            pred_events = predictions_by_file[fname]
            matches, fn, fp = match_events(gt_events, pred_events, args.tolerance)
            n_eval_files += 1

            for g, p, d in matches:
                matches_rows.append({"file": fname, **g, **p, "distance_m": round(d, 3)})
            for g in fn:
                fn_rows.append({"file": fname, **g})
            for p in fp:
                fp_rows.append({"file": fname, **p})
        except Exception as e:
            errors.append({"file": fname, "stage": "evaluation", "error": str(e)})
            print(f"[evaluation error] {fname}: {e}")

    matches_df = pd.DataFrame(matches_rows)
    fn_df = pd.DataFrame(fn_rows)
    fp_df = pd.DataFrame(fp_rows)
    matches_df.to_csv(out_dir / "test_matches.csv", index=False)
    fn_df.to_csv(out_dir / "test_false_negatives.csv", index=False)
    fp_df.to_csv(out_dir / "test_false_positives.csv", index=False)
    pd.DataFrame(errors).to_csv(out_dir / "test_errors.csv", index=False)

    event_classes = ["bend", "connector", "break"]
    # For event detection, TP is a matched event; precision denominator includes unmatched predicted events.
    per_class_rows = []
    for cls in event_classes:
        tp = int((matches_df["gt_type"] == cls).sum()) if not matches_df.empty else 0
        fn_count = int((fn_df["gt_type"] == cls).sum()) if not fn_df.empty else 0
        fp_count = int((fp_df["pred_type"] == cls).sum()) if not fp_df.empty else 0
        precision = tp / (tp + fp_count) if tp + fp_count else 0.0
        recall = tp / (tp + fn_count) if tp + fn_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_rows.append({"class": cls, "TP": tp, "FP": fp_count, "FN": fn_count,
                               "precision": precision, "recall": recall, "f1": f1})

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(out_dir / "test_per_class_metrics.csv", index=False)

    tp_total = int(per_class_df["TP"].sum())
    fp_total = int(per_class_df["FP"].sum())
    fn_total = int(per_class_df["FN"].sum())
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    summary = [
        "STRICT END-TO-END TEST", 
        "Prediction used CSV traces only; JSON masks were opened only after prediction for evaluation.",
        f"Test CSV files: {len(csv_paths)}",
        f"Test files with masks evaluated: {n_eval_files}",
        f"Prediction confidence threshold: {args.confidence}",
        f"Position matching tolerance: ±{args.tolerance} m",
        "",
        f"Overall event detection: TP={tp_total}, FP={fp_total}, FN={fn_total}",
        f"Overall precision={precision:.4f}",
        f"Overall recall={recall:.4f}",
        f"Overall F1={f1:.4f}",
        "",
        "Per class:",
        per_class_df.to_string(index=False),
        "",
        f"Prediction/evaluation errors: {len(errors)}",
    ]
    text = "\n".join(summary)
    (out_dir / "test_metrics_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nAll files saved in: {out_dir}")


if __name__ == "__main__":
    main()
