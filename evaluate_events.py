"""
evaluate_events.py — EVENT-LEVEL оценка OTDR-детектора.

ground truth  = клики в otdr_studio.py, сохранённые в .mask.json
                -> mask["annotations"] = [{"m":67.0,"db":...,"type":"connector"}, ...]
prediction    = события, которые находит детектор на трассе
                -> result["events"] = [{"m":67.0,"type":"connector","loss_dB":...}, ...]

Запуск:
    python evaluate_events.py --data "C:\\...\\test_dataset" --detector piecewise --tolerance 5
    python evaluate_events.py --data "C:\\...\\test_dataset" --detector piecewise --tolerance 5 --expected-m 150 --force-expected-m
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd

APP_DIR = r"C:\Users\Magzhan\OTDR_Classification\OTDR_Classification"
sys.path.append(APP_DIR)

from otdr_common import parse_otdr_csv, trim_dead_zone
from otdr_trends import analyze_trends
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def apply_smoothing(db, method, window):
    if method == "None" or window < 3:
        return db.astype(float)
    if method == "Moving average":
        w = _odd(window)
        return np.convolve(db, np.ones(w) / w, mode="same")
    if method == "Savitzky-Golay":
        w = _odd(min(window, len(db) - 1))
        return savgol_filter(db, window_length=max(5, w), polyorder=3, mode="interp")
    if method == "Median":
        return medfilt(db, kernel_size=_odd(window))
    if method == "Gaussian":
        return gaussian_filter1d(db, sigma=max(1.0, window / 6.0))
    return db.astype(float)


def get_predicted_events(km, db, db_s, zones, params, detector_name):
    if detector_name == "piecewise":
        out = analyze_trends(km, db, expected_length_m=params["expected_m"],
                              region_method=params["region_method"])
        return [{"m": float(e["m"]), "type": e["type"]} for e in out["events"]]

    elif detector_name == "trained":
        import detector_trained
        result = detector_trained.detect(km, db, db_s, zones, params)
        return [{"m": float(e["m"]), "type": e["type"]} for e in result["events"]]

    elif detector_name == "regression":
        from otdr_studio import detect_regression_channel
        result = detect_regression_channel(km, db, db_s, zones, params)
        return [{"m": float(e["m"]), "type": e["type"]} for e in result["events"]]

    raise ValueError(f"unknown detector {detector_name}")


def get_gt_events(mask):
    annos = mask.get("annotations", [])
    keep_types = {"bend", "break", "connector"}
    return [{"m": float(a["m"]), "type": a["type"]} for a in annos if a.get("type") in keep_types]


def match_events(gt_events, pred_events, tolerance_m, ignore_type):
    gt_left = list(enumerate(gt_events))
    pred_left = list(enumerate(pred_events))
    matches = []

    candidates = []
    for gi, g in gt_left:
        for pi, p in pred_left:
            d = abs(g["m"] - p["m"])
            if d <= tolerance_m and (ignore_type or g["type"] == p["type"]):
                candidates.append((d, gi, pi))
    candidates.sort()

    used_gt, used_pred = set(), set()
    for d, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gt_events[gi], pred_events[pi], d))

    fn_list = [g for gi, g in gt_left if gi not in used_gt]
    fp_list = [p for pi, p in pred_left if pi not in used_pred]
    return matches, fn_list, fp_list


def load_pairs(data_dir):
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    pairs = []
    for c in csvs:
        mask_path = os.path.splitext(c)[0] + ".mask.json"
        if os.path.exists(mask_path):
            pairs.append((c, mask_path))
    return pairs


def evaluate(data_dir, detector_name, tolerance_m, ignore_type,
             smooth_method, smooth_window, expected_m_default, region_method,
             force_expected_m):

    pairs = load_pairs(data_dir)
    if not pairs:
        raise SystemExit(f"Не найдено пар CSV + .mask.json в {data_dir}.")

    per_file_rows = []
    all_matches, all_fn, all_fp = [], [], []

    for csv_path, mask_path in pairs:
        fname = os.path.basename(csv_path)
        try:
            mask = json.load(open(mask_path, encoding="utf-8"))
            gt_events = get_gt_events(mask)

            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)
            db_s = apply_smoothing(db, smooth_method, smooth_window)

            zones = mask.get("ignore_zones_m", [])

            # ИСПРАВЛЕНО: --force-expected-m теперь реально работает.
            if force_expected_m:
                exp_m = expected_m_default
            else:
                exp_m = mask.get("gt_length_m") or mask.get("expected_length_m") or expected_m_default

            params = {"expected_m": exp_m, "sigma": 2.5, "break_frac": 0.85,
                      "bend_drop": 0.8, "region_method": region_method}

            pred_events = get_predicted_events(km, db, db_s, zones, params, detector_name)

            matches, fn_list, fp_list = match_events(gt_events, pred_events, tolerance_m, ignore_type)

            for g, p, d in matches:
                all_matches.append({"file": fname, "gt_type": g["type"], "pred_type": p["type"],
                                     "gt_m": g["m"], "pred_m": p["m"], "dist_m": round(d, 2),
                                     "type_correct": g["type"] == p["type"]})
            for g in fn_list:
                all_fn.append({"file": fname, "gt_type": g["type"], "gt_m": g["m"]})
            for p in fp_list:
                all_fp.append({"file": fname, "pred_type": p["type"], "pred_m": p["m"]})

            per_file_rows.append({
                "file": fname, "exp_m_used": exp_m, "n_gt": len(gt_events), "n_pred": len(pred_events),
                "n_matched": len(matches), "n_missed": len(fn_list), "n_extra": len(fp_list),
            })
        except Exception as e:
            print(f"[error] {fname}: {e}")

    return (pd.DataFrame(per_file_rows), pd.DataFrame(all_matches),
            pd.DataFrame(all_fn), pd.DataFrame(all_fp))


def compute_metrics(matches_df, fn_df, fp_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    n_tp = len(matches_df)
    n_fn = len(fn_df)
    n_fp = len(fp_df)

    detection_precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    detection_recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    detection_f1 = (2 * detection_precision * detection_recall /
                     (detection_precision + detection_recall)) if (detection_precision + detection_recall) else 0.0

    if not matches_df.empty:
        type_acc = matches_df["type_correct"].mean()
    else:
        type_acc = 0.0

    if matches_df.empty and fn_df.empty:
        per_type_df = pd.DataFrame(columns=["type", "gt_count", "precision", "recall", "f1"])
    else:
        gt_types = set(matches_df["gt_type"]) if not matches_df.empty else set()
        fn_types = set(fn_df["gt_type"]) if not fn_df.empty else set()
        all_types = sorted(gt_types.union(fn_types))
        per_type_rows = []
        for t in all_types:
            gt_t = ((matches_df["gt_type"] == t).sum() if not matches_df.empty else 0) + \
                   ((fn_df["gt_type"] == t).sum() if not fn_df.empty else 0)
            tp_t = ((matches_df["gt_type"] == t) & (matches_df["type_correct"])).sum() if not matches_df.empty else 0
            fp_as_t = (fp_df["pred_type"] == t).sum() if not fp_df.empty else 0
            pred_t = ((matches_df["pred_type"] == t).sum() if not matches_df.empty else 0) + fp_as_t
            prec_t = tp_t / pred_t if pred_t else 0.0
            rec_t = tp_t / gt_t if gt_t else 0.0
            f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) else 0.0
            per_type_rows.append({"type": t, "gt_count": int(gt_t), "precision": round(prec_t, 3),
                                   "recall": round(rec_t, 3), "f1": round(f1_t, 3)})
        per_type_df = pd.DataFrame(per_type_rows)

    matches_df.to_csv(os.path.join(out_dir, "event_matches.csv"), index=False)
    fn_df.to_csv(os.path.join(out_dir, "event_false_negatives.csv"), index=False)
    fp_df.to_csv(os.path.join(out_dir, "event_false_positives.csv"), index=False)
    per_type_df.to_csv(os.path.join(out_dir, "event_per_type_metrics.csv"), index=False)

    summary = [
        f"Detection (нашёл ли событие рядом с GT-точкой, независимо от типа):",
        f"  TP={n_tp}  FN(missed)={n_fn}  FP(extra)={n_fp}",
        f"  Precision={detection_precision:.3f}  Recall={detection_recall:.3f}  F1={detection_f1:.3f}",
        "",
        f"Type accuracy (среди совпавших по позиции, правильный ли тип): {type_acc:.3f}",
        "",
        "Per-type breakdown:",
        per_type_df.to_string(index=False),
        "",
        "Пропущенные события (false negatives) — алгоритм не нашёл вообще:",
        fn_df.to_string(index=False) if not fn_df.empty else "  (none)",
        "",
        "Лишние события (false positives) — алгоритм нашёл то, чего не было:",
        fp_df.to_string(index=False) if not fp_df.empty else "  (none)",
        "",
        "Совпавшие по позиции, но с неправильным типом:",
        matches_df[~matches_df["type_correct"]].to_string(index=False)
        if not matches_df.empty and (~matches_df["type_correct"]).any() else "  (none)",
    ]
    text = "\n".join(summary)
    with open(os.path.join(out_dir, "event_metrics_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description="Event-level оценка OTDR-детектора против ручных аннотаций.")
    ap.add_argument("--data", default=os.path.join(APP_DIR, "test_dataset"))
    ap.add_argument("--out", default=os.path.join(APP_DIR, "eval_events_results"))
    ap.add_argument("--detector", default="piecewise", choices=["piecewise", "trained", "regression"])
    ap.add_argument("--tolerance", type=float, default=5.0, help="допуск по метражу, метры")
    ap.add_argument("--ignore-type", action="store_true", help="считать матч только по позиции, без сверки типа")
    ap.add_argument("--smooth", default="Savitzky-Golay",
                    choices=["None", "Moving average", "Savitzky-Golay", "Median", "Gaussian"])
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--expected-m", type=float, default=125.0)
    ap.add_argument("--region-method", default="variance", choices=["variance", "level", "gradient"])
    ap.add_argument("--force-expected-m", action="store_true",
                    help="игнорировать gt_length_m/expected_length_m из маски, всегда брать --expected-m")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    per_file_df, matches_df, fn_df, fp_df = evaluate(
        args.data, args.detector, args.tolerance, args.ignore_type,
        args.smooth, args.window, args.expected_m, args.region_method,
        args.force_expected_m,
    )
    per_file_df.to_csv(os.path.join(args.out, "per_file_event_counts.csv"), index=False)
    compute_metrics(matches_df, fn_df, fp_df, args.out)


if __name__ == "__main__":
    main()