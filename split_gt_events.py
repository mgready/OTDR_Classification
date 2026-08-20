"""
split_gt_events.py — ПРАВИЛЬНЫЙ split + подготовка train/val датасета
напрямую из РУЧНЫХ annotations в .mask.json.

ВАЖНО: этот скрипт НЕ использует detect_breakpoints() для создания обучающих
примеров. В прошлом подходе detect_breakpoints находил только 355 из 1125
ручных аннотаций и почти терял класс break. Здесь КАЖДАЯ запись:

    {"m": 127.63, "db": -5.97, "type": "connector"}

из mask.json становится одной строкой в train_gt_events.csv или val_gt_events.csv.

Что делается:
  1. Ищет только пары .csv + .mask.json.
  2. Делит ФАЙЛЫ, а не события, на train/val (важно против data leakage).
  3. Для каждой ручной annotation вычисляет локальные signal features вокруг
     её точной позиции на трассе.
  4. Записывает только реальные 3 класса: bend, connector, break.
  5. Сохраняет:
     - split_files.json          : список файлов train/val (воспроизводимость)
     - train_gt_events.csv       : фичи + label для обучения
     - val_gt_events.csv         : фичи + label для честной валидации
     - split_summary.json        : статистика классов и файлов

Запуск:
  python split_gt_events.py --data "C:\\Users\\PCA\\OneDrive - Astana IT University\\Desktop\\OTDR_CLassification\\Dataset"

После этого обучай модель отдельным train_gt_models.py.
"""

import os
import sys
import glob
import json
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Скрипт предполагается положить рядом с otdr_common.py.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(APP_DIR)

from otdr_common import parse_otdr_csv, trim_dead_zone

CLASSES = ("bend", "connector", "break")

# Локальные числовые признаки. Они рассчитываются строго на точке ручной аннотации.
FEATURE_COLS = [
    "m_norm", "db_at_event", "local_mean_db", "local_std_db",
    "pre_mean_db", "post_mean_db", "loss_dB", "peak_above_bg_dB",
    "pre_slope_dB_per_km", "post_slope_dB_per_km", "slope_change_dB_per_km",
    "derivative_at_event_dB_per_km", "max_pre_derivative", "min_post_derivative",
    "pre_std_db", "post_std_db", "post_to_pre_std_ratio",
    "peak_width_m", "local_range_db",
]


def m_to_samples(km, meters, minimum=2):
    """Метры -> количество сэмплов с учётом реального spatial resolution CSV."""
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
    """Ширина локального подъёма на половине высоты; 0 для непикового события."""
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


def extract_features(km, db, annotation_m, expected_length_m=None):
    """Создаёт локальные признаки вокруг ручной GT-точки annotation_m."""
    idx = int(np.argmin(np.abs(km * 1000.0 - annotation_m)))
    n = len(db)

    # Окна до/после события. Можно менять позже, но не нужно для первого запуска.
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

    # Фоновый уровень для измерения отражающего пика.
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
        "m": round(float(km[idx] * 1000.0), 3),
        "sample_idx": idx,
        "m_norm": round(float((km[idx] * 1000.0) / nominal_m), 6),
        "db_at_event": round(float(db[idx]), 4),
        "local_mean_db": round(local_mean, 4),
        "local_std_db": round(local_std, 4),
        "pre_mean_db": round(pre_mean, 4),
        "post_mean_db": round(post_mean, 4),
        "loss_dB": round(float(loss), 4),
        "peak_above_bg_dB": round(float(peak_above_bg), 4),
        "pre_slope_dB_per_km": round(pre_slope, 4),
        "post_slope_dB_per_km": round(post_slope, 4),
        "slope_change_dB_per_km": round(post_slope - pre_slope, 4),
        "derivative_at_event_dB_per_km": round(float(grad[idx]), 4),
        "max_pre_derivative": round(float(np.max(pre_grad)) if len(pre_grad) else 0.0, 4),
        "min_post_derivative": round(float(np.min(post_grad)) if len(post_grad) else 0.0, 4),
        "pre_std_db": round(pre_std, 4),
        "post_std_db": round(post_std, 4),
        "post_to_pre_std_ratio": round(float(post_std / (pre_std + 1e-6)), 4),
        "peak_width_m": round(peak_width(km, db, idx, bg, peak_above_bg), 4),
        "local_range_db": round(float(np.ptp(local)) if len(local) else 0.0, 4),
    }


def find_labeled_pairs(data_dir):
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    pairs, skipped = [], 0
    for csv_path in csv_paths:
        mask_path = os.path.splitext(csv_path)[0] + ".mask.json"
        if os.path.exists(mask_path):
            pairs.append((csv_path, mask_path))
        else:
            skipped += 1
    print(f"[info] CSV всего: {len(csv_paths)} | с маской: {len(pairs)} | без маски: {skipped}")
    return pairs


def get_events_from_mask(mask_path):
    """Читает ТОЛЬКО реальные ручные классы из annotations."""
    data = json.load(open(mask_path, encoding="utf-8"))
    annotations = data.get("annotations", [])
    events = []
    for a in annotations:
        event_type = str(a.get("type", "")).strip().lower()
        if event_type in CLASSES and "m" in a:
            events.append({"m": float(a["m"]), "type": event_type})
    expected_length_m = data.get("gt_length_m") or data.get("expected_length_m")
    return events, expected_length_m


def file_stratum(mask_path):
    """Страта файла по комбинации классов — для более честного train/val split."""
    try:
        events, _ = get_events_from_mask(mask_path)
        types = sorted({e["type"] for e in events})
        return "+".join(types) if types else "none"
    except Exception:
        return "none"


def split_by_file(pairs, val_size=0.25, seed=42):
    idx = list(range(len(pairs)))
    strata = [file_stratum(mask_path) for _, mask_path in pairs]

    # Страты с 1 примером sklearn не может разделить; объединяем их.
    counts = Counter(strata)
    strata_safe = [s if counts[s] >= 2 else "rare" for s in strata]

    try:
        train_idx, val_idx = train_test_split(
            idx, test_size=val_size, random_state=seed, stratify=strata_safe
        )
    except ValueError as e:
        print(f"[warning] Не удалось стратифицировать split ({e}). Использую random file split.")
        train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=seed)

    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


def build_gt_table(pairs):
    """Создаёт таблицу: одна реальная annotation из JSON = одна строка."""
    rows = []
    failures = []

    for csv_path, mask_path in pairs:
        filename = os.path.basename(csv_path)
        try:
            gt_events, expected_length_m = get_events_from_mask(mask_path)
            if not gt_events:
                continue

            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)

            for event_no, event in enumerate(gt_events):
                row = extract_features(km, db, event["m"], expected_length_m)
                row.update({
                    "file": filename,
                    "event_no_in_file": event_no,
                    "gt_m": event["m"],
                    "label": event["type"],
                    "expected_length_m": expected_length_m,
                })
                rows.append(row)
        except Exception as e:
            failures.append({"file": filename, "error": str(e)})
            print(f"[error] {filename}: {e}")

    return pd.DataFrame(rows), failures


def count_labels(df):
    if df.empty:
        return {}
    return {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}


def main():
    ap = argparse.ArgumentParser(description="File-level train/val split из ручных JSON-аннотаций.")
    ap.add_argument("--data", required=True, help="папка Dataset с CSV + .mask.json")
    ap.add_argument("--out", default=os.path.join(APP_DIR, "gt_event_model_data"),
                    help="папка результатов")
    ap.add_argument("--val-size", type=float, default=0.25, help="доля validation файлов")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pairs = find_labeled_pairs(args.data)
    if not pairs:
        raise SystemExit("Не найдено ни одной пары CSV + .mask.json. Проверь --data.")

    train_pairs, val_pairs = split_by_file(pairs, args.val_size, args.seed)
    print(f"[info] Train файлов: {len(train_pairs)} | Val файлов: {len(val_pairs)}")

    print("\n--- Извлечение GT-событий: TRAIN ---")
    train_df, train_errors = build_gt_table(train_pairs)
    print("\n--- Извлечение GT-событий: VALIDATION ---")
    val_df, val_errors = build_gt_table(val_pairs)

    # Защита: никаких иных классов кроме трёх реальных не должно быть.
    train_df = train_df[train_df["label"].isin(CLASSES)].copy()
    val_df = val_df[val_df["label"].isin(CLASSES)].copy()

    train_csv = os.path.join(args.out, "train_gt_events.csv")
    val_csv = os.path.join(args.out, "val_gt_events.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    split_json = {
        "seed": args.seed,
        "val_size": args.val_size,
        "classes": list(CLASSES),
        "train_files": [os.path.basename(p[0]) for p in train_pairs],
        "val_files": [os.path.basename(p[0]) for p in val_pairs],
    }
    with open(os.path.join(args.out, "split_files.json"), "w", encoding="utf-8") as f:
        json.dump(split_json, f, ensure_ascii=False, indent=2)

    summary = {
        "n_labeled_files": len(pairs),
        "n_train_files": len(train_pairs),
        "n_val_files": len(val_pairs),
        "n_train_events": int(len(train_df)),
        "n_val_events": int(len(val_df)),
        "train_class_counts": count_labels(train_df),
        "val_class_counts": count_labels(val_df),
        "n_read_errors": len(train_errors) + len(val_errors),
        "read_errors": train_errors + val_errors,
        "feature_columns": FEATURE_COLS,
    }
    with open(os.path.join(args.out, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== ГОТОВО ===")
    print(f"Train: {len(train_df)} GT-событий | {count_labels(train_df)}")
    print(f"Val:   {len(val_df)} GT-событий | {count_labels(val_df)}")
    print(f"Всего: {len(train_df) + len(val_df)} GT-событий")
    print(f"Сохранено: {train_csv}")
    print(f"Сохранено: {val_csv}")
    print(f"Сохранено: {os.path.join(args.out, 'split_files.json')}")
    print(f"Сохранено: {os.path.join(args.out, 'split_summary.json')}")


if __name__ == "__main__":
    main()
