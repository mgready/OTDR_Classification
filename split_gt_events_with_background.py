"""
split_gt_events_with_background.py — готовит ЧЕСТНЫЙ 4-class event dataset:

  bend / connector / break  = только ручные annotations из .mask.json
  background                = автоматические отрицательные точки, гарантированно
                              далёкие от любой ручной annotation.

Зачем background:
  В прошлом 3-class обучении модель всегда была вынуждена назвать любой
  просканированный пик bend/connector/break. На end-to-end test это дало
  1106 false positives. Класс background позволяет модели сказать:
  "это не реальное событие".

Ключевые правила против leakage:
  1. СНАЧАЛА делим CSV-файлы на train/val.
  2. Потом из train-файлов создаём train positives+background;
     из val-файлов — val positives+background независимо.
  3. JSON используется только для подготовки истинных labels, не для
     будущего inference на test_dataset.

Background sampling:
  - candidate-like negative: пик/степ, найденный по CSV, но дальше
    --exclusion-m от любой ручной GT точки;
  - stable random negative: точка в спокойной зоне трассы, тоже далеко от GT.
  - max_background_per_positive ограничивает баланс классов.

Запуск:
  python split_gt_events_with_background.py --data "C:\\Users\\PCA\\OneDrive - Astana IT University\\Desktop\\OTDR_CLassification\\Dataset"
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split

APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))
from otdr_common import parse_otdr_csv, trim_dead_zone

POS_CLASSES = ("bend", "connector", "break")
ALL_CLASSES = ("bend", "connector", "break", "background")

FEATURE_COLS = [
    "m_norm", "db_at_event", "local_mean_db", "local_std_db",
    "pre_mean_db", "post_mean_db", "loss_dB", "peak_above_bg_dB",
    "pre_slope_dB_per_km", "post_slope_dB_per_km", "slope_change_dB_per_km",
    "derivative_at_event_dB_per_km", "max_pre_derivative", "min_post_derivative",
    "pre_std_db", "post_std_db", "post_to_pre_std_ratio",
    "peak_width_m", "local_range_db",
]


def m_to_samples(km, meters, minimum=2):
    if len(km) < 2:
        return minimum
    dx = float(np.median(np.diff(km)))
    return max(minimum, int((meters / 1000.0) / dx)) if dx > 0 else minimum


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
        "sample_idx": int(idx),
        "m_norm": round(float((km[idx] * 1000.0) / nominal_m), 6),
        "db_at_event": round(float(db[idx]), 4),
        "local_mean_db": round(local_mean, 4),
        "local_std_db": round(local_std, 4),
        "pre_mean_db": round(pre_mean, 4),
        "post_mean_db": round(post_mean, 4),
        "loss_dB": round(float(pre_mean - post_mean), 4),
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


def moving_average(x, samples):
    samples = max(3, int(samples) | 1)
    if len(x) < samples:
        return x.astype(float)
    return np.convolve(x, np.ones(samples) / samples, mode="same")


def candidate_indices(km, db):
    """Та же high-recall candidate generation, что будет использована на test."""
    n = len(db)
    if n < 10:
        return []
    sm = moving_average(db, m_to_samples(km, 1.0))
    dist = m_to_samples(km, 1.5)
    baseline = np.median(sm)
    mad = np.median(np.abs(sm - baseline)) + 1e-6

    peaks, _ = find_peaks(sm, prominence=max(0.7, 1.5 * mad), distance=dist)

    step = np.zeros(n)
    for i in range(dist, n - dist):
        step[i] = np.median(sm[i - dist:i]) - np.median(sm[i:i + dist])
    step_thr = max(0.25, np.percentile(step[dist:n-dist], 85)) if n > 2 * dist else 0.25
    down, _ = find_peaks(step, height=step_thr, distance=dist)

    mins, _ = find_peaks(-sm, prominence=max(0.25, 0.5 * mad), distance=dist)
    guard = m_to_samples(km, 5.0)
    return sorted(set(int(i) for i in np.concatenate([peaks, down, mins]) if guard <= i < n - guard))


def get_events_from_mask(mask_path):
    data = json.load(open(mask_path, encoding="utf-8"))
    events = []
    for a in data.get("annotations", []):
        typ = str(a.get("type", "")).strip().lower()
        if typ in POS_CLASSES and "m" in a:
            events.append({"m": float(a["m"]), "type": typ})
    exp_m = data.get("gt_length_m") or data.get("expected_length_m")
    return events, exp_m


def find_pairs(data_dir):
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    pairs = []
    for csv_path in csvs:
        mask_path = os.path.splitext(csv_path)[0] + ".mask.json"
        if os.path.exists(mask_path):
            pairs.append((csv_path, mask_path))
    print(f"[info] CSV total={len(csvs)} | labeled CSV+mask pairs={len(pairs)} | without mask={len(csvs)-len(pairs)}")
    return pairs


def stratum(mask_path):
    try:
        ev, _ = get_events_from_mask(mask_path)
        x = sorted({e["type"] for e in ev})
        return "+".join(x) if x else "none"
    except Exception:
        return "none"


def split_files(pairs, val_size, seed):
    idx = list(range(len(pairs)))
    st = [stratum(m) for _, m in pairs]
    counts = Counter(st)
    st = [x if counts[x] >= 2 else "rare" for x in st]
    try:
        tr, va = train_test_split(idx, test_size=val_size, random_state=seed, stratify=st)
    except ValueError:
        tr, va = train_test_split(idx, test_size=val_size, random_state=seed)
    return [pairs[i] for i in tr], [pairs[i] for i in va]


def far_from_gt(m, gt_meters, exclusion_m):
    return all(abs(m - g) > exclusion_m for g in gt_meters)


def make_rows(pairs, rng, background_ratio, exclusion_m, stable_ratio):
    """Одна GT annotation -> positive row; controlled negatives -> background rows."""
    rows, errors = [], []

    for csv_path, mask_path in pairs:
        fname = os.path.basename(csv_path)
        try:
            gt_events, exp_m = get_events_from_mask(mask_path)
            if not gt_events:
                continue
            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)
            gt_meters = [e["m"] for e in gt_events]

            # Все POSITIVE examples: именно и только ручные annotations.
            for event_no, event in enumerate(gt_events):
                idx = int(np.argmin(np.abs(km * 1000.0 - event["m"])))
                row = extract_features(km, db, idx, exp_m)
                row.update({"file": fname, "source": "manual_annotation",
                            "event_no_in_file": event_no, "gt_m": event["m"],
                            "label": event["type"]})
                rows.append(row)

            # Background budget ограничен числом реальных событий на этом файле.
            n_bg_target = max(1, int(np.ceil(len(gt_events) * background_ratio)))
            n_candidate_target = int(round(n_bg_target * (1.0 - stable_ratio)))
            n_stable_target = n_bg_target - n_candidate_target

            # 1. Hard negatives: detector-like candidates, но далеко от всех GT.
            candidates = candidate_indices(km, db)
            hard = [i for i in candidates if far_from_gt(float(km[i] * 1000.0), gt_meters, exclusion_m)]
            rng.shuffle(hard)
            selected = hard[:n_candidate_target]

            # 2. Stable negatives: случайные легкие background точки далеко от GT.
            guard = m_to_samples(km, 5.0)
            valid_idx = np.array([i for i in range(guard, len(km) - guard)
                                  if far_from_gt(float(km[i] * 1000.0), gt_meters, exclusion_m)])
            if len(valid_idx):
                rng.shuffle(valid_idx)
                for i in valid_idx:
                    if len(selected) >= n_candidate_target + n_stable_target:
                        break
                    # Не берём почти ту же точку, что уже добавили как hard-negative.
                    if all(abs(km[i] * 1000.0 - km[j] * 1000.0) > exclusion_m for j in selected):
                        selected.append(int(i))

            for bg_no, idx in enumerate(selected):
                row = extract_features(km, db, idx, exp_m)
                row.update({"file": fname,
                            "source": "hard_negative" if idx in hard[:n_candidate_target] else "stable_background",
                            "event_no_in_file": bg_no,
                            "gt_m": np.nan,
                            "label": "background"})
                rows.append(row)

        except Exception as e:
            errors.append({"file": fname, "error": str(e)})
            print(f"[error] {fname}: {e}")

    return pd.DataFrame(rows), errors


def counts_dict(df):
    return {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()} if not df.empty else {}


def main():
    ap = argparse.ArgumentParser(description="Build 4-class GT+background file-level train/val datasets.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(APP_DIR / "gt_event_model_data_4class"))
    ap.add_argument("--val-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--background-ratio", type=float, default=2.0,
                    help="background examples per positive event, default=2")
    ap.add_argument("--exclusion-m", type=float, default=8.0,
                    help="background must be farther than this from every GT annotation")
    ap.add_argument("--stable-ratio", type=float, default=0.25,
                    help="fraction of background from stable random points; remaining are hard negatives")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_train = np.random.default_rng(args.seed)
    rng_val = np.random.default_rng(args.seed + 1)

    pairs = find_pairs(args.data)
    if not pairs:
        raise SystemExit("No CSV+mask pairs found.")
    train_pairs, val_pairs = split_files(pairs, args.val_size, args.seed)
    print(f"[info] File split: train={len(train_pairs)}, val={len(val_pairs)}")

    print("\n--- TRAIN: manual positives + controlled background ---")
    train_df, train_errors = make_rows(train_pairs, rng_train, args.background_ratio,
                                       args.exclusion_m, args.stable_ratio)
    print("\n--- VALIDATION: manual positives + controlled background ---")
    val_df, val_errors = make_rows(val_pairs, rng_val, args.background_ratio,
                                   args.exclusion_m, args.stable_ratio)

    train_csv = out_dir / "train_4class_events.csv"
    val_csv = out_dir / "val_4class_events.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    with open(out_dir / "split_files.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "train_files": [os.path.basename(x[0]) for x in train_pairs],
                   "val_files": [os.path.basename(x[0]) for x in val_pairs]}, f, indent=2, ensure_ascii=False)

    summary = {
        "classes": list(ALL_CLASSES),
        "background_definition": "candidate/random point farther than exclusion_m from every manual annotation",
        "background_ratio": args.background_ratio,
        "exclusion_m": args.exclusion_m,
        "stable_ratio": args.stable_ratio,
        "train_files": len(train_pairs), "val_files": len(val_pairs),
        "train_rows": len(train_df), "val_rows": len(val_df),
        "train_counts": counts_dict(train_df), "val_counts": counts_dict(val_df),
        "errors": train_errors + val_errors,
        "features": FEATURE_COLS,
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== READY ===")
    print(f"Train: {len(train_df)} rows | {counts_dict(train_df)}")
    print(f"Val:   {len(val_df)} rows | {counts_dict(val_df)}")
    print(f"Saved: {train_csv}")
    print(f"Saved: {val_csv}")
    print(f"Saved: {out_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
