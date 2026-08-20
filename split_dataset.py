"""
split_dataset_3class.py — версия split_dataset.py БЕЗ синтетического класса
"noise". Обучающая выборка строится СТРОГО из твоих трёх классов
(bend/connector/break), как ты их размечал в otdr_studio.py.

Отличие от split_dataset.py: breakpoint-кандидаты, для которых НЕ нашлось
пары среди твоих annotations (в пределах --tolerance метров), просто
ВЫБРАСЫВАЮТСЯ из обучающей выборки, а не помечаются как "noise".

Важно понимать: итоговая модель будет уметь классифицировать НАЙДЕННЫЙ
кандидат как bend/connector/break, но не сможет сказать "это не событие" —
на инференсе она всегда выберет один из трёх классов для любого пика/степа,
который найдёт detect_breakpoints, даже если это случайный шум. Если тебе
важно, чтобы модель умела отбрасывать ложные срабатывания — используй
исходный split_dataset.py (с классом noise) вместо этого файла.

Запуск:
    python split_dataset_3class.py --data "C:\\...\\Dataset"
"""

import os
import sys
import glob
import json
import argparse
from collections import Counter
import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(APP_DIR)

from otdr_common import parse_otdr_csv, trim_dead_zone
from otdr_trends import find_valid_region, detect_breakpoints, _m_to_samples
from sklearn.model_selection import train_test_split


FEATURE_COLS = ["loss_dB", "reflectance_dB", "is_reflective", "width_m",
                "frac_of_expected", "dist_to_eof_m", "post_signal_std",
                "post_signal_mean_db", "pre_slope_db_per_km", "post_slope_db_per_km"]


def local_features(km, db, sm, i, start, eof, w, expected_length_m, refl_set):
    n = len(db)
    pre = np.median(sm[max(0, i - w):i]) if i > start else sm[i]
    post = np.median(sm[i:min(n, i + w)]) if i < eof else sm[i]
    loss_db = float(pre - post)
    reflectance_db = float(sm[max(0, i - w):min(n, i + w)].max() - max(pre, post))
    is_reflective = int(i in refl_set)

    if is_reflective:
        peak_val = sm[i]
        half = peak_val - reflectance_db / 2.0
        lo = i
        while lo > 0 and sm[lo] > half:
            lo -= 1
        hi = i
        while hi < n - 1 and sm[hi] > half:
            hi += 1
        width_m = float((km[hi] - km[lo]) * 1000.0)
    else:
        width_m = 0.0

    frac_of_expected = float((km[i] * 1000.0) / expected_length_m) if expected_length_m else np.nan
    dist_to_eof_m = float((km[eof] - km[i]) * 1000.0)

    post_w = _m_to_samples(km, 8.0)
    post_seg = db[i:min(n, i + post_w)]
    post_signal_std = float(post_seg.std()) if len(post_seg) > 2 else 0.0
    post_signal_mean_db = float(post_seg.mean()) if len(post_seg) > 0 else float(post)

    pre_w = max(3, _m_to_samples(km, 5.0))
    pre_seg_km, pre_seg_db = km[max(start, i - pre_w):i], sm[max(start, i - pre_w):i]
    post_seg_km, post_seg_db = km[i:min(eof, i + pre_w)], sm[i:min(eof, i + pre_w)]
    pre_slope = float(np.polyfit(pre_seg_km, pre_seg_db, 1)[0]) if len(pre_seg_km) > 2 else 0.0
    post_slope = float(np.polyfit(post_seg_km, post_seg_db, 1)[0]) if len(post_seg_km) > 2 else 0.0

    return {
        "loss_dB": round(loss_db, 3),
        "reflectance_dB": round(reflectance_db, 3),
        "is_reflective": is_reflective,
        "width_m": round(width_m, 3),
        "frac_of_expected": round(frac_of_expected, 4) if not np.isnan(frac_of_expected) else 0.0,
        "dist_to_eof_m": round(dist_to_eof_m, 3),
        "post_signal_std": round(post_signal_std, 3),
        "post_signal_mean_db": round(post_signal_mean_db, 3),
        "pre_slope_db_per_km": round(pre_slope, 3),
        "post_slope_db_per_km": round(post_slope, 3),
    }


def get_gt_events(mask):
    annos = mask.get("annotations", [])
    keep = {"bend", "break", "connector"}
    return [{"m": float(a["m"]), "type": a["type"]} for a in annos if a.get("type") in keep]


def find_labeled_files(data_dir):
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    labeled = []
    skipped = 0
    for c in csvs:
        mask_path = os.path.splitext(c)[0] + ".mask.json"
        if os.path.exists(mask_path):
            labeled.append((c, mask_path))
        else:
            skipped += 1
    print(f"[info] CSV всего: {len(csvs)} | с маской: {len(labeled)} | без маски (пропущены): {skipped}")
    return labeled


def file_stratum(mask_path):
    try:
        mask = json.load(open(mask_path, encoding="utf-8"))
        types = sorted({a.get("type") for a in mask.get("annotations", [])
                        if a.get("type") in {"bend", "break", "connector"}})
        return "+".join(types) if types else "none"
    except Exception:
        return "none"


def split_files(labeled_pairs, val_size=0.25, seed=42):
    strata = [file_stratum(mp) for _, mp in labeled_pairs]
    counts = Counter(strata)
    strata = [s if counts[s] >= 2 else "rare" for s in strata]

    idx = list(range(len(labeled_pairs)))
    try:
        train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=seed,
                                               stratify=strata)
    except ValueError:
        train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=seed)

    train_pairs = [labeled_pairs[i] for i in train_idx]
    val_pairs = [labeled_pairs[i] for i in val_idx]
    return train_pairs, val_pairs


def build_events_df(pairs, region_method="variance", expected_m_default=150.0,
                     tolerance_m=5.0):
    """
    ГЛАВНОЕ ОТЛИЧИЕ от split_dataset.py: кандидаты БЕЗ пары в GT просто
    пропускаются (continue), а не добавляются в датасет с лейблом "noise".
    Итоговый df содержит СТРОГО только строки с label in {bend, connector, break}.
    """
    rows = []
    n_matched_total, n_unmatched_total = 0, 0

    for csv_path, mask_path in pairs:
        fname = os.path.basename(csv_path)
        try:
            mask = json.load(open(mask_path, encoding="utf-8"))
            gt_events = get_gt_events(mask)
            if not gt_events:
                continue

            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)
            exp_m = mask.get("gt_length_m") or mask.get("expected_length_m") or expected_m_default

            start, eof = find_valid_region(km, db, method=region_method)
            bps, refl_set, sm = detect_breakpoints(km, db, start, eof)
            w = _m_to_samples(km, 1.5)

            gt_used = set()
            for i in bps:
                m_i = float(km[i] * 1000.0)
                best_j, best_d = None, tolerance_m + 1
                for j, g in enumerate(gt_events):
                    if j in gt_used:
                        continue
                    d = abs(g["m"] - m_i)
                    if d <= tolerance_m and d < best_d:
                        best_j, best_d = j, d

                if best_j is None:
                    n_unmatched_total += 1
                    continue  # <-- КЛЮЧЕВОЕ ОТЛИЧИЕ: пропускаем, не лейблим "noise"

                label = gt_events[best_j]["type"]
                gt_used.add(best_j)
                n_matched_total += 1

                feat = local_features(km, db, sm, i, start, eof, w, exp_m, refl_set)
                feat.update({"file": fname, "m": round(m_i, 2), "label": label})
                rows.append(feat)
        except Exception as e:
            print(f"[error] {fname}: {e}")

    print(f"[info] Совпало с GT (использовано в датасете): {n_matched_total}")
    print(f"[info] Кандидатов без пары (ВЫБРОШЕНО, не noise): {n_unmatched_total}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Split датасета на train/val + сборка признаков, СТРОГО 3 класса.")
    ap.add_argument("--data", required=True, help="папка с CSV + .mask.json")
    ap.add_argument("--out", default=os.path.join(APP_DIR, "event_model_results_3class"))
    ap.add_argument("--val-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--region-method", default="variance", choices=["variance", "level", "gradient"])
    ap.add_argument("--expected-m", type=float, default=150.0)
    ap.add_argument("--tolerance", type=float, default=5.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    labeled_pairs = find_labeled_files(args.data)
    if len(labeled_pairs) == 0:
        raise SystemExit("Не найдено ни одной пары CSV+.mask.json — проверь --data путь.")

    train_pairs, val_pairs = split_files(labeled_pairs, val_size=args.val_size, seed=args.seed)
    print(f"Train файлов: {len(train_pairs)} | Val файлов: {len(val_pairs)}")

    with open(os.path.join(args.out, "split_files.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train": [os.path.basename(c) for c, _ in train_pairs],
            "val": [os.path.basename(c) for c, _ in val_pairs],
        }, f, indent=2, ensure_ascii=False)

    print("\n--- building TRAIN events ---")
    train_df = build_events_df(train_pairs, args.region_method, args.expected_m, args.tolerance)
    print("\n--- building VAL events ---")
    val_df = build_events_df(val_pairs, args.region_method, args.expected_m, args.tolerance)

    train_df.to_csv(os.path.join(args.out, "train_events.csv"), index=False)
    val_df.to_csv(os.path.join(args.out, "val_events.csv"), index=False)

    print(f"\nTrain events: {len(train_df)} rows | class counts:")
    print(train_df["label"].value_counts() if not train_df.empty else "empty")
    print(f"\nVal events: {len(val_df)} rows | class counts:")
    print(val_df["label"].value_counts() if not val_df.empty else "empty")
    print(f"\nSaved -> {os.path.join(args.out, 'train_events.csv')}")
    print(f"Saved -> {os.path.join(args.out, 'val_events.csv')}")


if __name__ == "__main__":
    main()