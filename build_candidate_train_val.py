"""
build_candidate_train_val.py — ЧЕСТНАЯ подготовка candidate-level train/validation.

Проблема старого подхода:
  Положительные признаки считались в точной ручной GT-координате. Поэтому
  validation F1 ~0.99 измерял только "как хорошо назвать событие, если точка
  уже известна", а не реальную задачу поиска событий на трассе.

ЭТОТ СКРИПТ строит train и validation одинаково:

  CSV -> candidate generator (JSON НЕ используется) -> local features ->
  JSON открывается только ПОСЛЕ этого, чтобы присвоить candidate label:
     same-type/near GT not needed for label assignment:
       candidate within tolerance of ANY GT -> GT class (bend/connector/break)
       candidate away from all GT -> background

Ключевое:
  - СНАЧАЛА split по ФАЙЛАМ train/val.
  - train CSV candidates + train JSON labels -> train_candidates.csv.
  - val CSV candidates + val JSON labels -> val_candidates.csv.
  - Модель позже обучается ТОЛЬКО train_candidates.csv.
  - val_candidates.csv используется ТОЛЬКО для model selection/threshold tuning.
  - Внутри одного CSV кандидаты создаются без JSON — одинаково для train/val/test.

Запуск:
  python build_candidate_train_val.py --data "C:\\Users\\PCA\\OneDrive - Astana IT University\\Desktop\\OTDR_CLassification\\Dataset"
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

EVENT_CLASSES = ("bend", "connector", "break")
ALL_CLASSES = EVENT_CLASSES + ("background",)
FEATURE_COLS = [
    "m_norm", "db_at_event", "local_mean_db", "local_std_db",
    "pre_mean_db", "post_mean_db", "loss_dB", "peak_above_bg_dB",
    "pre_slope_dB_per_km", "post_slope_dB_per_km", "slope_change_dB_per_km",
    "derivative_at_event_dB_per_km", "max_pre_derivative", "min_post_derivative",
    "pre_std_db", "post_std_db", "post_to_pre_std_ratio", "peak_width_m",
    "local_range_db",
]


def m_to_samples(km, meters, minimum=2):
    if len(km) < 2:
        return minimum
    dx = float(np.median(np.diff(km)))
    return max(minimum, int((meters / 1000.0) / dx)) if dx > 0 else minimum


def moving_average(x, samples):
    samples = max(3, int(samples) | 1)
    if len(x) < samples:
        return x.astype(float)
    return np.convolve(x, np.ones(samples) / samples, mode="same")


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
    """CANDIDATES ONLY FROM CSV. No JSON or GT is referenced in this function."""
    n = len(db)
    if n < 10:
        return []

    sm = moving_average(db, m_to_samples(km, 1.0))
    distance = m_to_samples(km, 1.5)
    baseline = np.median(sm)
    mad = np.median(np.abs(sm - baseline)) + 1e-6

    peaks, _ = find_peaks(sm, prominence=max(0.7, 1.5 * mad), distance=distance)

    # Downward step candidate score.
    step = np.zeros(n)
    for i in range(distance, n - distance):
        step[i] = np.median(sm[i-distance:i]) - np.median(sm[i:i+distance])
    threshold = max(0.25, np.percentile(step[distance:n-distance], 85)) if n > 2 * distance else 0.25
    down, _ = find_peaks(step, height=threshold, distance=distance)

    # Local minima catch terminal drops / break-like points.
    mins, _ = find_peaks(-sm, prominence=max(0.25, 0.5 * mad), distance=distance)

    guard = m_to_samples(km, 5.0)
    idxs = np.unique(np.concatenate([peaks, down, mins])).astype(int)
    return sorted(i for i in idxs if guard <= i < n - guard)


def extract_features(km, db, idx, expected_length_m=None):
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
        "m": round(float(km[idx] * 1000.0), 3),
        "candidate_idx": int(idx),
        "m_norm": round(float(km[idx] * 1000.0 / nominal), 6),
        "db_at_event": round(float(db[idx]), 4),
        "local_mean_db": round(safe_mean(local, db[idx]), 4),
        "local_std_db": round(safe_std(local), 4),
        "pre_mean_db": round(pre, 4),
        "post_mean_db": round(post, 4),
        "loss_dB": round(float(pre-post), 4),
        "peak_above_bg_dB": round(float(peak_height), 4),
        "pre_slope_dB_per_km": round(pre_slope, 4),
        "post_slope_dB_per_km": round(post_slope, 4),
        "slope_change_dB_per_km": round(post_slope-pre_slope, 4),
        "derivative_at_event_dB_per_km": round(float(grad[idx]), 4),
        "max_pre_derivative": round(float(np.max(pre_grad)) if len(pre_grad) else 0.0, 4),
        "min_post_derivative": round(float(np.min(post_grad)) if len(post_grad) else 0.0, 4),
        "pre_std_db": round(safe_std(left), 4),
        "post_std_db": round(safe_std(right), 4),
        "post_to_pre_std_ratio": round(float(safe_std(right)/(safe_std(left)+1e-6)), 4),
        "peak_width_m": round(peak_width(km, db, idx, bg, peak_height), 4),
        "local_range_db": round(float(np.ptp(local)) if len(local) else 0.0, 4),
    }


def get_gt(mask_path):
    """Reads human annotation only AFTER candidate generation has occurred."""
    data = json.load(open(mask_path, encoding="utf-8"))
    events = []
    for a in data.get("annotations", []):
        typ = str(a.get("type", "")).strip().lower()
        if typ in EVENT_CLASSES and "m" in a:
            events.append({"m": float(a["m"]), "type": typ})
    exp_m = data.get("gt_length_m") or data.get("expected_length_m")
    return events, exp_m


def assign_labels(candidates_m, gt_events, tolerance_m):
    """After CSV-only candidate generation, assign each candidate a GT label.

    One-to-one nearest matching prevents one GT event from labeling multiple
    candidates. Remaining candidates are background.
    """
    possible = []
    for ci, m in enumerate(candidates_m):
        for gi, g in enumerate(gt_events):
            d = abs(m - g["m"])
            if d <= tolerance_m:
                possible.append((d, ci, gi))
    possible.sort()

    used_c, used_g = set(), set()
    labels = ["background"] * len(candidates_m)
    gt_m_for_candidate = [np.nan] * len(candidates_m)
    for d, ci, gi in possible:
        if ci in used_c or gi in used_g:
            continue
        used_c.add(ci)
        used_g.add(gi)
        labels[ci] = gt_events[gi]["type"]
        gt_m_for_candidate[ci] = gt_events[gi]["m"]

    missed_gt = [g for gi, g in enumerate(gt_events) if gi not in used_g]
    return labels, gt_m_for_candidate, missed_gt


def find_pairs(data_dir):
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    pairs = []
    for csv in csvs:
        mask = os.path.splitext(csv)[0] + ".mask.json"
        if os.path.exists(mask):
            pairs.append((csv, mask))
    print(f"[info] CSV total={len(csvs)} | CSV+mask={len(pairs)} | no mask={len(csvs)-len(pairs)}")
    return pairs


def stratum(mask_path):
    try:
        ev, _ = get_gt(mask_path)
        types = sorted({x["type"] for x in ev})
        return "+".join(types) if types else "none"
    except Exception:
        return "none"


def split_files(pairs, val_size, seed):
    idx = list(range(len(pairs)))
    st = [stratum(mask) for _, mask in pairs]
    c = Counter(st)
    st = [x if c[x] >= 2 else "rare" for x in st]
    try:
        tr, va = train_test_split(idx, test_size=val_size, random_state=seed, stratify=st)
    except ValueError:
        tr, va = train_test_split(idx, test_size=val_size, random_state=seed)
    return [pairs[i] for i in tr], [pairs[i] for i in va]


def build_candidate_table(pairs, tolerance_m, split_name):
    rows, missed_rows, errors = [], [], []
    total_candidates = 0

    for number, (csv_path, mask_path) in enumerate(pairs, 1):
        fname = os.path.basename(csv_path)
        try:
            # 1) Candidate generation from CSV ONLY.
            km, db, _ = parse_otdr_csv(csv_path)
            km, db = trim_dead_zone(km, db)
            idxs = candidate_indices_from_csv(km, db)
            candidate_m = [float(km[i] * 1000.0) for i in idxs]
            total_candidates += len(idxs)

            # 2) JSON is opened only NOW to assign labels to already fixed candidates.
            gt_events, expected_m = get_gt(mask_path)
            labels, gt_m_values, missed_gt = assign_labels(candidate_m, gt_events, tolerance_m)

            for idx, label, gt_m in zip(idxs, labels, gt_m_values):
                feat = extract_features(km, db, idx, expected_m)
                feat.update({"file": fname, "split": split_name, "label": label, "gt_m": gt_m})
                rows.append(feat)

            for event in missed_gt:
                missed_rows.append({"file": fname, "split": split_name,
                                    "gt_m": event["m"], "gt_type": event["type"]})

        except Exception as e:
            errors.append({"file": fname, "error": str(e)})
            print(f"[error] {fname}: {e}")

        if number % 50 == 0 or number == len(pairs):
            print(f"[{split_name}] {number}/{len(pairs)} files processed")

    df = pd.DataFrame(rows)
    missed = pd.DataFrame(missed_rows)
    print(f"[{split_name}] candidates={total_candidates} | rows={len(df)} | "
          f"label counts={df['label'].value_counts().to_dict() if not df.empty else {}} | "
          f"GT events missed by candidate generator={len(missed)}")
    return df, missed, errors


def main():
    ap = argparse.ArgumentParser(description="Build honest candidate-level train/validation datasets.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(APP_DIR / "candidate_event_model_data"))
    ap.add_argument("--val-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tolerance", type=float, default=5.0,
                    help="candidate-GT match tolerance for candidate labels")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(args.data)
    if not pairs:
        raise SystemExit("No CSV+mask pairs found.")

    train_pairs, val_pairs = split_files(pairs, args.val_size, args.seed)
    print(f"[info] File split: train={len(train_pairs)}, validation={len(val_pairs)}")

    print("\n--- Build TRAIN candidates: CSV candidates first -> JSON labels after ---")
    train_df, train_missed, train_errors = build_candidate_table(train_pairs, args.tolerance, "train")
    print("\n--- Build VALIDATION candidates: CSV candidates first -> JSON labels after ---")
    val_df, val_missed, val_errors = build_candidate_table(val_pairs, args.tolerance, "validation")

    train_df.to_csv(out / "train_candidates.csv", index=False)
    val_df.to_csv(out / "val_candidates.csv", index=False)
    train_missed.to_csv(out / "train_gt_missed_by_candidates.csv", index=False)
    val_missed.to_csv(out / "val_gt_missed_by_candidates.csv", index=False)

    with open(out / "split_files.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "tolerance_m": args.tolerance,
                   "train_files": [os.path.basename(x[0]) for x in train_pairs],
                   "val_files": [os.path.basename(x[0]) for x in val_pairs]}, f, indent=2, ensure_ascii=False)

    summary = {
        "method": "CSV-only candidates; JSON labels assigned after candidates are fixed",
        "classes": list(ALL_CLASSES),
        "train_files": len(train_pairs), "val_files": len(val_pairs),
        "train_rows": len(train_df), "val_rows": len(val_df),
        "train_counts": train_df["label"].value_counts().to_dict() if not train_df.empty else {},
        "val_counts": val_df["label"].value_counts().to_dict() if not val_df.empty else {},
        "train_gt_missed_by_candidate_generator": len(train_missed),
        "val_gt_missed_by_candidate_generator": len(val_missed),
        "errors": train_errors + val_errors,
        "feature_columns": FEATURE_COLS,
    }
    with open(out / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== DONE ===")
    print(f"Saved: {out / 'train_candidates.csv'}")
    print(f"Saved: {out / 'val_candidates.csv'}")
    print(f"Saved: {out / 'build_summary.json'}")
    print("IMPORTANT: candidate generator misses are structural recall ceiling; inspect *_gt_missed_by_candidates.csv.")


if __name__ == "__main__":
    main()
