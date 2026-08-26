"""
make_article_figures_clean.py — clean MDPI-style figures for the OTDR paper.

Creates:
  Figure_4_matched_event_confusion_matrix.png
  Figure_5a_connector_break.png
  Figure_5b_multiple_events.png
  Figure_5c_bend.png
  Figure_5d_error_case.png
  Figure_6_per_class_metrics_clean.png
  Table_4_test_metrics_clean.csv
  figure_manifest_clean.txt

Improvements:
  - no "Pred." prefix in labels;
  - prediction and GT labels are placed in separate vertical lanes;
  - solid lines = model output; dashed lines = manual reference annotation;
  - Figure 4 is a standard 3x3 class-agreement matrix for matched events;
  - false positives/false negatives are reported in Table 4, not mixed into matrix.

Run:
  python make_article_figures_clean.py --data "C:\\Users\\PCA\\OneDrive - Astana IT University\\Desktop\\OTDR_CLassification\\test_dataset"
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
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))
from otdr_common import parse_otdr_csv, trim_dead_zone

DEFAULT_MODEL = APP_DIR / "gt_event_model_data_4class" / "model_results" / "best_4class_event_classifier.joblib"
DEFAULT_OUT = APP_DIR / "article_figures_clean"
EVENT_CLASSES = ["bend", "connector", "break"]
COLORS = {"bend": "#D55E00", "connector": "#009E73", "break": "#0072B2"}


def setup_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "axes.linewidth": 0.8,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })


def m_to_samples(km, meters, minimum=2):
    if len(km) < 2:
        return minimum
    dx = float(np.median(np.diff(km)))
    return max(minimum, int((meters / 1000.0) / dx)) if dx > 0 else minimum


def smooth(x, window):
    window = max(3, int(window) | 1)
    if len(x) < window:
        return x.astype(float)
    return np.convolve(x, np.ones(window) / window, mode="same")


def safe_mean(x, fallback=0.0):
    return float(np.mean(x)) if len(x) else float(fallback)


def safe_std(x):
    return float(np.std(x)) if len(x) > 1 else 0.0


def local_slope(x, y):
    if len(x) < 3 or np.ptp(x) <= 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def peak_width(km, db, idx, bg, height):
    if height <= 0:
        return 0.0
    half = bg + height / 2.0
    lo = hi = idx
    while lo > 0 and db[lo] > half:
        lo -= 1
    while hi < len(db) - 1 and db[hi] > half:
        hi += 1
    return float((km[hi] - km[lo]) * 1000.0)


def extract_features(km, db, idx, expected_m=None):
    n = len(db)
    ws, wc, wsl = m_to_samples(km, 1.5), m_to_samples(km, 6.0), m_to_samples(km, 5.0)
    left = db[max(0, idx-wc):idx]
    right = db[idx+1:min(n, idx+1+wc)]
    local = db[max(0, idx-ws):min(n, idx+ws+1)]
    pre, post = safe_mean(left, db[idx]), safe_mean(right, db[idx])
    bg_values = np.concatenate([left, right]) if len(left)+len(right) else np.array([db[idx]])
    bg = float(np.median(bg_values))
    height = float(np.max(local) - bg)
    pre_slope = local_slope(km[max(0,idx-wsl):idx], db[max(0,idx-wsl):idx])
    post_slope = local_slope(km[idx+1:min(n,idx+1+wsl)], db[idx+1:min(n,idx+1+wsl)])
    grad = np.gradient(db, km) if n > 2 else np.zeros(n)
    pre_grad = grad[max(0,idx-ws):idx+1]
    post_grad = grad[idx:min(n,idx+ws+1)]
    nominal = float(expected_m) if expected_m and expected_m > 0 else float(km[-1]*1000.0)
    return {
        "m_norm": float(km[idx]*1000.0/nominal),
        "db_at_event": float(db[idx]),
        "local_mean_db": safe_mean(local, db[idx]),
        "local_std_db": safe_std(local),
        "pre_mean_db": pre,
        "post_mean_db": post,
        "loss_dB": pre-post,
        "peak_above_bg_dB": height,
        "pre_slope_dB_per_km": pre_slope,
        "post_slope_dB_per_km": post_slope,
        "slope_change_dB_per_km": post_slope-pre_slope,
        "derivative_at_event_dB_per_km": float(grad[idx]),
        "max_pre_derivative": float(np.max(pre_grad)) if len(pre_grad) else 0.0,
        "min_post_derivative": float(np.min(post_grad)) if len(post_grad) else 0.0,
        "pre_std_db": safe_std(left),
        "post_std_db": safe_std(right),
        "post_to_pre_std_ratio": safe_std(right)/(safe_std(left)+1e-6),
        "peak_width_m": peak_width(km, db, idx, bg, height),
        "local_range_db": float(np.ptp(local)) if len(local) else 0.0,
    }


def candidate_indices(km, db):
    n = len(db)
    sm = smooth(db, m_to_samples(km, 1.0))
    distance = m_to_samples(km, 1.5)
    baseline = np.median(sm)
    mad = np.median(np.abs(sm-baseline)) + 1e-6
    peaks, _ = find_peaks(sm, prominence=max(0.7, 1.5*mad), distance=distance)
    step = np.zeros(n)
    for i in range(distance, n-distance):
        step[i] = np.median(sm[i-distance:i]) - np.median(sm[i:i+distance])
    threshold = max(0.25, np.percentile(step[distance:n-distance], 85)) if n > 2*distance else 0.25
    down, _ = find_peaks(step, height=threshold, distance=distance)
    mins, _ = find_peaks(-sm, prominence=max(0.25, 0.5*mad), distance=distance)
    guard = m_to_samples(km, 5.0)
    idxs = np.unique(np.concatenate([peaks, down, mins])).astype(int)
    return idxs[(idxs >= guard) & (idxs < n-guard)]


def nms(events, radius=5.0):
    out = []
    for event in sorted(events, key=lambda x: x["confidence"], reverse=True):
        if all(abs(event["m"] - old["m"]) > radius for old in out):
            out.append(event)
    return sorted(out, key=lambda x: x["m"])


def predict_csv_only(csv_path, bundle, confidence):
    km, db, _ = parse_otdr_csv(csv_path)
    km, db = trim_dead_zone(km, db)
    idxs = candidate_indices(km, db)
    model, feature_names = bundle["model"], bundle["features"]
    if len(idxs) == 0:
        return km, db, []

    rows = [extract_features(km, db, int(i)) for i in idxs]
    X = pd.DataFrame([{f: row[f] for f in feature_names} for row in rows], columns=feature_names)
    labels = model.predict(X).astype(str)
    probas = model.predict_proba(X)
    classes = [str(c) for c in model.classes_]

    events = []
    for idx, label, proba in zip(idxs, labels, probas):
        if label not in EVENT_CLASSES:
            continue
        conf = float(proba[classes.index(label)])
        if conf >= confidence:
            events.append({"m": float(km[idx]*1000.0), "type": label, "confidence": conf})
    return km, db, nms(events)


def load_gt(mask_path):
    if not os.path.exists(mask_path):
        return []
    data = json.load(open(mask_path, encoding="utf-8"))
    return [{"m": float(a["m"]), "type": a["type"]}
            for a in data.get("annotations", []) if a.get("type") in EVENT_CLASSES]


def match_events(gt, pred, tolerance):
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g["type"] == p["type"]:
                dist = abs(g["m"] - p["m"])
                if dist <= tolerance:
                    candidates.append((dist, gi, pi))
    candidates.sort()
    used_gt, used_pred, matches = set(), set(), []
    for dist, gi, pi in candidates:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi); used_pred.add(pi)
            matches.append((gt[gi], pred[pi], dist))
    fn = [g for i, g in enumerate(gt) if i not in used_gt]
    fp = [p for i, p in enumerate(pred) if i not in used_pred]
    return matches, fn, fp


def plot_trace(km, db, pred, gt, title, path, dpi):
    m = km*1000.0
    sm = smooth(db, m_to_samples(km, 1.0))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(m, db, color="#B7D0E8", lw=0.65, label="Raw trace")
    ax.plot(m, sm, color="#244F70", lw=1.15, label="Smoothed trace")
    top = max(np.max(db), np.max(sm)) + 1.5
    bottom = min(np.min(db), -35.0)
    ax.set_ylim(bottom, top)

    # Separate label lanes prevent overlap. Solid=prediction; dashed=reference.
    pred_lane = {"bend": 0.91, "connector": 0.79, "break": 0.67}
    gt_lane = {"bend": 0.10, "connector": 0.20, "break": 0.30}

    for event in pred:
        color = COLORS[event["type"]]
        ax.axvline(event["m"], color=color, lw=1.6, ls="-", alpha=0.95)
        ax.annotate(f"{event['type']}\n{event['m']:.1f} m",
                    xy=(event["m"], pred_lane[event["type"]]), xycoords=("data", "axes fraction"),
                    ha="center", va="center", fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, alpha=0.95))

    for event in gt:
        color = COLORS[event["type"]]
        ax.axvline(event["m"], color=color, lw=1.15, ls="--", alpha=0.9)
        ax.annotate(f"{event['type']}\n{event['m']:.1f} m",
                    xy=(event["m"], gt_lane[event["type"]]), xycoords=("data", "axes fraction"),
                    ha="center", va="center", fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.18", fc="#F7F7F7", ec=color, alpha=0.95))

    handles = [
        mpl.lines.Line2D([], [], color="#B7D0E8", lw=1, label="Raw trace"),
        mpl.lines.Line2D([], [], color="#244F70", lw=1.3, label="Smoothed trace"),
        mpl.lines.Line2D([], [], color="black", lw=1.5, ls="-", label="Model output"),
        mpl.lines.Line2D([], [], color="black", lw=1.2, ls="--", label="Reference annotation"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, ncol=2)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Signal level (dB)")
    ax.set_title(title, pad=6)
    ax.grid(alpha=0.22, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_confusion(matches, path, dpi):
    matrix = np.zeros((3, 3), dtype=int)
    for gt, pred, _ in matches:
        matrix[EVENT_CLASSES.index(gt["type"]), EVENT_CLASSES.index(pred["type"])] += 1

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), ["Bend", "Connector", "Break"])
    ax.set_yticks(range(3), ["Bend", "Connector", "Break"])
    ax.set_xlabel("Predicted event class")
    ax.set_ylabel("Reference event class")
    ax.set_title("Matched event classes — independent test")
    vmax = max(matrix.max(), 1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if matrix[i, j] > vmax*0.55 else "black")
    fig.colorbar(image, ax=ax, label="Number of matched events", shrink=0.84)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_metrics(metrics, path, dpi):
    x = np.arange(len(metrics)); width = 0.24
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.bar(x-width, metrics["precision"], width, label="Precision", color="#0072B2")
    ax.bar(x, metrics["recall"], width, label="Recall", color="#009E73")
    ax.bar(x+width, metrics["f1"], width, label="F1 score", color="#D55E00")
    ax.set_xticks(x, [c.capitalize() for c in metrics["class"]])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Independent test performance by event class")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Create clean article-ready OTDR figures.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--confidence", type=float, default=0.45)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    setup_style()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model)
    csv_paths = sorted(glob.glob(os.path.join(args.data, "*.csv")))

    records, all_matches, all_fn, all_fp = [], [], [], []
    for n, csv_path in enumerate(csv_paths, 1):
        filename = os.path.basename(csv_path)
        km, db, pred = predict_csv_only(csv_path, bundle, args.confidence)
        gt = load_gt(os.path.splitext(csv_path)[0] + ".mask.json")
        matches, fn, fp = match_events(gt, pred, args.tolerance)
        records.append({"file": filename, "km": km, "db": db, "pred": pred,
                        "gt": gt, "matches": matches, "fn": fn, "fp": fp})
        all_matches.extend(matches)
        all_fn.extend([{"file": filename, **x} for x in fn])
        all_fp.extend([{"file": filename, **x} for x in fp])
        print(f"[{n}/{len(csv_paths)}] {filename}: predictions={len(pred)}, GT={len(gt)}, matched={len(matches)}")

    metrics = []
    for cls in EVENT_CLASSES:
        tp = sum(g["type"] == cls and p["type"] == cls for g, p, _ in all_matches)
        fn = sum(x["type"] == cls for x in all_fn)
        fp = sum(x["type"] == cls for x in all_fp)
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        metrics.append({"class": cls, "TP": tp, "FP": fp, "FN": fn,
                        "precision": precision, "recall": recall, "f1": f1})
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "Table_4_test_metrics_clean.csv", index=False)

    plot_confusion(all_matches, out_dir / "Figure_4_matched_event_confusion_matrix.png", args.dpi)
    plot_metrics(metrics_df, out_dir / "Figure_6_per_class_metrics_clean.png", args.dpi)

    def choose(predicate, fallback_index):
        found = [r for r in records if predicate(r)]
        return found[0] if found else records[fallback_index]

    selected = {
        "Figure_5a_connector_break": choose(
            lambda r: len(r["matches"]) >= 2 and {"connector", "break"}.issubset({g["type"] for g, _, _ in r["matches"]}), 0),
        "Figure_5b_multiple_events": choose(
            lambda r: len(r["gt"]) >= 3 and len(r["matches"]) >= 2, min(1, len(records)-1)),
        "Figure_5c_bend": choose(
            lambda r: any(g["type"] == "bend" for g, _, _ in r["matches"]), min(2, len(records)-1)),
        "Figure_5d_error_case": max(records, key=lambda r: len(r["fn"]) + len(r["fp"])),
    }

    manifest = []
    for fig_name, record in selected.items():
        path = out_dir / f"{fig_name}.png"
        plot_trace(record["km"], record["db"], record["pred"], record["gt"],
                   f"{fig_name.replace('_', ' ')} — {record['file']}", path, args.dpi)
        manifest.append(f"{fig_name}.png | source={record['file']} | predictions={len(record['pred'])} | "
                        f"GT={len(record['gt'])} | matched={len(record['matches'])} | "
                        f"FN={len(record['fn'])} | FP={len(record['fp'])}")

    manifest.extend([
        "Figure_4_matched_event_confusion_matrix.png | 3x3 matrix of matched event classes",
        "Figure_6_per_class_metrics_clean.png | class-specific precision, recall, and F1",
        "Table_4_test_metrics_clean.csv | numerical independent-test metrics",
    ])
    (out_dir / "figure_manifest_clean.txt").write_text("\n".join(manifest), encoding="utf-8")
    print(f"\nGenerated clean figures in: {out_dir}")


if __name__ == "__main__":
    main()
