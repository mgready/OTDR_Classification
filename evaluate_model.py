"""
evaluate_model.py — сравнение detector_trained.py (обученная модель) с
ручными лейблами (.mask.json), сделанными в otdr_studio.py.

ИСПРАВЛЕНО (v2): раньше в модель передавался НЕсглаженный сигнал (db вместо db_s),
из-за чего classifier почти всегда предсказывал "break" (100% файлов).
Теперь смузинг применяется ТОЧНО как в otdr_studio.py (Studio._reanalyze),
и expected_m по умолчанию берётся из самого joblib-бандла модели (b["expected_m"]),
а не из ручной разметки, если явно не указано другое.

Запуск:
    python evaluate_model.py --data "C:\\Users\\Magzhan\\OTDR_Classification\\OTDR_Classification\\test_dataset"

Опции:
    --smooth {None,Moving average,Savitzky-Golay,Median,Gaussian}  (default: Savitzky-Golay)
    --window <int>                                                  (default: 15)
    --use-model-expected-m / --use-gt-expected-m   (какую длину трассы передавать в модель)
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d

# ── подключаем те же модули, что использует otdr_studio.py ──────────────────
APP_DIR = r"C:\Users\Magzhan\OTDR_Classification\OTDR_Classification"
sys.path.append(APP_DIR)

from otdr_common import parse_otdr_csv, trim_dead_zone
import detector_trained  # твой custom detector (detector_trained.py)


CLASS_NAME_FROM_VERDICT = {
    "NORMAL": "normal",
    "BEND": "bend",
    "CONNECTOR": "connector",
    "BREAK": "break",
}
NUM_TO_NAME = {1: "normal", 2: "bend", 3: "break", 4: "connector"}


# ── та же функция сглаживания, что в otdr_studio.py ──────────────────────────
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


def parse_pred_name(result: dict) -> str:
    verdict = result.get("verdict", "")
    for key, name in CLASS_NAME_FROM_VERDICT.items():
        if key in verdict.upper():
            return name
    feats = result.get("features", {})
    p_items = {k[2:]: v for k, v in feats.items() if k.startswith("p_")}
    if p_items:
        return max(p_items, key=p_items.get)
    return "unknown"


def parse_pred_proba(result: dict, pred_name: str) -> float:
    feats = result.get("features", {})
    return feats.get(f"p_{pred_name}", np.nan)


def normalize_gt(gt_raw):
    if gt_raw is None:
        return None
    if isinstance(gt_raw, (int, float)):
        return NUM_TO_NAME.get(int(gt_raw), str(gt_raw))
    s = str(gt_raw).strip().lower()
    if s.isdigit():
        return NUM_TO_NAME.get(int(s), s)
    return s


def load_pairs(data_dir):
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    pairs = []
    for c in csvs:
        mask_path = os.path.splitext(c)[0] + ".mask.json"
        if os.path.exists(mask_path):
            pairs.append((c, mask_path))
    return pairs


def evaluate(data_dir, smooth_method, smooth_window, use_model_expected_m):
    bundle = detector_trained._bundle()
    model_expected_m = bundle.get("expected_m", 150.0)
    print(f"[info] model trained with expected_m = {model_expected_m}")
    print(f"[info] model expects features: {bundle.get('features')}")
    print(f"[info] model classes: {bundle.get('names')}")

    pairs = load_pairs(data_dir)
    if not pairs:
        raise SystemExit(
            f"Не найдено пар CSV + .mask.json в {data_dir}. "
            f"Сначала разметь трассы в otdr_studio.py и сохрани маски (Save mask)."
        )

    rows = []
    for csv_path, mask_path in pairs:
        fname = os.path.basename(csv_path)
        try:
            mask = json.load(open(mask_path, encoding="utf-8"))
            gt = normalize_gt(mask.get("gt_class"))
            if gt is None:
                print(f"[skip] {fname}: в маске нет gt_class")
                continue

            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)

            # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: то же сглаживание, что в GUI по умолчанию
            db_s = apply_smoothing(db, smooth_method, smooth_window)

            zones = mask.get("ignore_zones_m", [])

            exp_m = model_expected_m if use_model_expected_m else \
                (mask.get("gt_length_m") or mask.get("expected_length_m") or model_expected_m)

            p = {
                "expected_m": exp_m,
                "sigma": 2.5,
                "break_frac": 0.85,
                "bend_drop": 0.8,
                "region_method": "variance",
            }

            result = detector_trained.detect(km, db, db_s, zones, p)
            pred = parse_pred_name(result)
            proba = parse_pred_proba(result, pred)

            rows.append({
                "file": fname,
                "gt_class": gt,
                "pred_class": pred,
                "proba": proba,
                "verdict": result.get("verdict", ""),
                "features": json.dumps(result.get("features", {})),
                "correct": gt == pred,
            })
        except Exception as e:
            print(f"[error] {fname}: {e}")

    if not rows:
        raise SystemExit("Не удалось получить ни одного валидного предсказания.")

    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    y_true = df["gt_class"].values
    y_pred = df["pred_class"].values
    labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = pd.DataFrame({
        "class": labels, "precision": precision, "recall": recall,
        "f1": f1, "support": support,
    })

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                          columns=[f"pred_{l}" for l in labels])

    per_class.to_csv(os.path.join(out_dir, "metrics_report.csv"), index=False)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    report_txt = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    summary_lines = [
        f"N samples: {len(df)}",
        f"Accuracy: {acc:.4f}",
        f"Macro    precision/recall/F1: {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}",
        f"Weighted precision/recall/F1: {weighted_p:.4f} / {weighted_r:.4f} / {weighted_f1:.4f}",
        "", "Per-class:", per_class.to_string(index=False),
        "", "Confusion matrix (rows=true, cols=pred):", cm_df.to_string(),
        "", "sklearn classification_report:", report_txt,
        "", "Misclassified files:",
        df[~df["correct"]][["file", "gt_class", "pred_class", "verdict"]].to_string(index=False)
        if (~df["correct"]).any() else "  (none — 100% accuracy)",
    ]
    summary = "\n".join(summary_lines)
    with open(os.path.join(out_dir, "metrics_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "per_class": per_class, "confusion_matrix": cm_df}


def main():
    ap = argparse.ArgumentParser(description="Оценка OTDR-классификатора против ручных лейблов.")
    ap.add_argument("--data", default=os.path.join(APP_DIR, "test_dataset"))
    ap.add_argument("--out", default=os.path.join(APP_DIR, "eval_results"))
    ap.add_argument("--smooth", default="Savitzky-Golay",
                    choices=["None", "Moving average", "Savitzky-Golay", "Median", "Gaussian"])
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--use-model-expected-m", dest="use_model_expected_m", action="store_true", default=True)
    ap.add_argument("--use-gt-expected-m", dest="use_model_expected_m", action="store_false")
    args = ap.parse_args()

    df = evaluate(args.data, args.smooth, args.window, args.use_model_expected_m)
    compute_metrics(df, args.out)


if __name__ == "__main__":
    main()