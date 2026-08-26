"""
make_article_figures.py — publication-ready figures for the OTDR event-detection paper.

Creates high-resolution PNG figures from the FINAL test results produced by
`test_4class_event_model_fast.py`:

  figures/Figure_4_test_confusion_matrix.png
  figures/Figure_5a_correct_connector_break.png
  figures/Figure_5b_multiple_events.png
  figures/Figure_5c_bend_case.png
  figures/Figure_5d_error_case.png
  figures/Figure_6_per_class_metrics.png
  figures/Table_4_test_metrics.csv
  figures/figure_manifest.txt

The script predicts from CSV only. It reads JSON masks only after prediction,
for reference overlays and for selecting representative examples. Dashed lines
are GT, solid lines are model predictions.

Usage:
  python make_article_figures.py --data "C:\\Users\\PCA\\OneDrive - Astana IT University\\Desktop\\OTDR_CLassification\\test_dataset"

Optional:
  python make_article_figures.py --data "...\\test_dataset" --dpi 600 --confidence 0.45

Expected model (default):
  gt_event_model_data_4class/model_results/best_4class_event_classifier.joblib

Expected test output (optional, but useful):
  gt_event_model_data_4class/test_results_fast/test_matches_4class_fast.csv
  gt_event_model_data_4class/test_results_fast/test_false_negatives_4class_fast.csv
  gt_event_model_data_4class/test_results_fast/test_false_positives_4class_fast.csv
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
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))
from otdr_common import parse_otdr_csv, trim_dead_zone

DEFAULT_MODEL = APP_DIR / "gt_event_model_data_4class" / "model_results" / "best_4class_event_classifier.joblib"
DEFAULT_RESULTS = APP_DIR / "gt_event_model_data_4class" / "test_results_fast"
DEFAULT_OUT = APP_DIR / "article_figures"
EVENT_CLASSES = {"bend", "connector", "break"}
PRED_COLORS = {"bend": "#E69F00", "connector": "#009E73", "break": "#D55E00"}
GT_COLORS = {"bend": "#8C5A00", "connector": "#006A50", "break": "#8B1A1A"}


def configure_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def m_to_samples(km, meters, minimum=2):
    if len(km) < 2:
        return minimum
    dx = float(np.median(np.diff(km)))
    return max(minimum, int((meters / 1000.0) / dx)) if dx > 0 else minimum


def moving_average(x, samples):
    samples = max(3, int(samples) | 1)
    return np.convolve(x, np.ones(samples) / samples, mode="same") if len(x) >= samples else x.astype(float)


def safe_mean(x, fallback=0.0):
    return float(np.mean(x)) if len(x) else float(fallback)


def safe_std(x):
    return float(np.std(x)) if len(x) > 1 else 0.0


def local_slope(x, y):
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 3 and np.ptp(x) > 0 else 0.0


def peak_width(km, db, idx, bg, height):
    if height <= 0:
        return 0.0
    half = bg + height / 2.0
    lo, hi = idx, idx
    while lo > 0 and db[lo] > half:
        lo -= 1
    while hi < len(db) - 1 and db[hi] > half:
        hi += 1
    return float((km[hi] - km[lo]) * 1000.0)


def candidates(km, db):
    n = len(db)
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
    return idxs[(idxs >= guard) & (idxs < n-guard)]


def extract_features(km, db, idx, expected_m=None):
    n = len(db)
    ws, wc, wsl = m_to_samples(km, 1.5), m_to_samples(km, 6.0), m_to_samples(km, 5.0)
    left = db[max(0, idx-wc):idx]
    right = db[idx+1:min(n, idx+1+wc)]
    local = db[max(0, idx-ws):min(n, idx+ws+1)]
    pre, post = safe_mean(left, db[idx]), safe_mean(right, db[idx])
    bg_values = np.concatenate([left, right]) if len(left)+len(right) else np.array([db[idx]])
    bg = float(np.median(bg_values))
    ph = float(np.max(local)) - bg
    pre_slope = local_slope(km[max(0, idx-wsl):idx], db[max(0, idx-wsl):idx])
    post_slope = local_slope(km[idx+1:min(n, idx+1+wsl)], db[idx+1:min(n, idx+1+wsl)])
    grad = np.gradient(db, km) if n > 2 else np.zeros(n)
    pre_grad = grad[max(0, idx-ws):idx+1]
    post_grad = grad[idx:min(n, idx+ws+1)]
    nominal = float(expected_m) if expected_m and expected_m > 0 else float(km[-1]*1000.0)
    return {
        "m_norm": float(km[idx]*1000.0/nominal), "db_at_event": float(db[idx]),
        "local_mean_db": safe_mean(local, db[idx]), "local_std_db": safe_std(local),
        "pre_mean_db": pre, "post_mean_db": post, "loss_dB": pre-post,
        "peak_above_bg_dB": ph, "pre_slope_dB_per_km": pre_slope,
        "post_slope_dB_per_km": post_slope, "slope_change_dB_per_km": post_slope-pre_slope,
        "derivative_at_event_dB_per_km": float(grad[idx]),
        "max_pre_derivative": float(np.max(pre_grad)) if len(pre_grad) else 0.0,
        "min_post_derivative": float(np.min(post_grad)) if len(post_grad) else 0.0,
        "pre_std_db": safe_std(left), "post_std_db": safe_std(right),
        "post_to_pre_std_ratio": safe_std(right)/(safe_std(left)+1e-6),
        "peak_width_m": peak_width(km, db, idx, bg, ph),
        "local_range_db": float(np.ptp(local)) if len(local) else 0.0,
    }


def nms(events, radius_m=5.0):
    kept = []
    for event in sorted(events, key=lambda x: x["confidence"], reverse=True):
        if all(abs(event["m"] - old["m"]) > radius_m for old in kept):
            kept.append(event)
    return sorted(kept, key=lambda x: x["m"])


def predict_csv_only(csv_path, bundle, threshold):
    km, db, _ = parse_otdr_csv(csv_path)
    km, db = trim_dead_zone(km, db)
    idxs = candidates(km, db)
    model, features = bundle["model"], bundle["features"]
    classes = [str(c) for c in model.classes_]

    feature_rows = [extract_features(km, db, int(i)) for i in idxs]
    if not feature_rows:
        return km, db, []
    X = pd.DataFrame([{f: row[f] for f in features} for row in feature_rows], columns=features)
    pred = model.predict(X).astype(str)
    proba = model.predict_proba(X)

    events = []
    for idx, row, lab, probs in zip(idxs, feature_rows, pred, proba):
        if lab not in EVENT_CLASSES:
            continue
        conf = float(probs[classes.index(lab)])
        if conf >= threshold:
            events.append({"m": float(km[idx]*1000.0), "type": lab, "confidence": conf})
    return km, db, nms(events, radius_m=5.0)


def load_gt(mask_path):
    if not os.path.exists(mask_path):
        return []
    d = json.load(open(mask_path, encoding="utf-8"))
    return [{"m": float(a["m"]), "type": a["type"]}
            for a in d.get("annotations", []) if a.get("type") in EVENT_CLASSES]


def match(gt, pred, tolerance=5.0):
    pairs = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g["type"] == p["type"]:
                d = abs(g["m"] - p["m"])
                if d <= tolerance:
                    pairs.append((d, gi, pi))
    pairs.sort()
    used_g, used_p, matches = set(), set(), []
    for d, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi); used_p.add(pi); matches.append((gt[gi], pred[pi], d))
    return matches, [g for i,g in enumerate(gt) if i not in used_g], [p for i,p in enumerate(pred) if i not in used_p]


def plot_trace(km, db, pred, gt, title, out_path, dpi):
    m = km * 1000.0
    smooth = moving_average(db, m_to_samples(km, 1.0))
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.plot(m, db, color="#A7C7E7", linewidth=0.65, alpha=0.8, label="Raw OTDR trace")
    ax.plot(m, smooth, color="#1B4F72", linewidth=1.1, label="Smoothed trace")

    ylim_top = max(np.max(db), np.max(smooth)) + 1.5
    ylim_bottom = min(np.min(db), -35)
    ax.set_ylim(ylim_bottom, ylim_top)

    # Solid = model output. Draw labels on top.
    for e in pred:
        color = PRED_COLORS[e["type"]]
        ax.axvline(e["m"], color=color, lw=1.6, ls="-", zorder=4)
        ax.annotate(f"Pred. {e['type']}\n{e['m']:.1f} m\n{e['confidence']:.2f}",
                    xy=(e["m"], ylim_top), xytext=(0, -3), textcoords="offset points",
                    ha="center", va="top", fontsize=7.2, color=color,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, alpha=0.95),
                    clip_on=True)

    # Dashed = GT. Labels placed from bottom to avoid overlap.
    for e in gt:
        color = GT_COLORS[e["type"]]
        ax.axvline(e["m"], color=color, lw=1.2, ls="--", zorder=3)
        ax.annotate(f"GT {e['type']}\n{e['m']:.1f} m", xy=(e["m"], ylim_bottom),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.0, color=color,
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=color, alpha=0.9),
                    clip_on=True)

    handles = [
        mpl.lines.Line2D([], [], color="#A7C7E7", lw=1.2, label="Raw OTDR trace"),
        mpl.lines.Line2D([], [], color="#1B4F72", lw=1.3, label="Smoothed trace"),
        mpl.lines.Line2D([], [], color="black", lw=1.4, ls="-", label="Model prediction"),
        mpl.lines.Line2D([], [], color="black", lw=1.2, ls="--", label="Ground truth"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Signal level (dB)")
    ax.set_title(title, pad=7)
    ax.grid(alpha=0.22, lw=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_confusion(matches_df, fn_df, fp_df, out_path, dpi):
    classes = ["bend", "connector", "break"]
    # Matrix rows true events; columns: predicted class / missed. Include FPs separately in annotation.
    matrix = np.zeros((3, 4), dtype=int)
    for _, r in matches_df.iterrows():
        matrix[classes.index(r["gt_type"]), classes.index(r["pred_type"])] += 1
    for _, r in fn_df.iterrows():
        matrix[classes.index(r["type"]), 3] += 1

    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(4), ["Pred. bend", "Pred. connector", "Pred. break", "Missed"])
    ax.set_yticks(range(3), ["GT bend", "GT connector", "GT break"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > matrix.max()*0.55 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(im, ax=ax, label="Number of events", shrink=0.84)
    fp_text = ", ".join(f"{c}: {(fp_df['type']==c).sum()}" if not fp_df.empty else f"{c}: 0" for c in classes)
    ax.set_title(f"Independent test: event matching matrix\nFalse positives: {fp_text}")
    ax.set_xlabel("Detection outcome")
    ax.set_ylabel("Reference event class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_metrics(metrics_df, out_path, dpi):
    classes = metrics_df["class"].tolist()
    x = np.arange(len(classes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    ax.bar(x-width, metrics_df["precision"], width, label="Precision", color="#0072B2")
    ax.bar(x, metrics_df["recall"], width, label="Recall", color="#009E73")
    ax.bar(x+width, metrics_df["f1"], width, label="F1 score", color="#D55E00")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, [c.capitalize() for c in classes])
    ax.set_ylabel("Score")
    ax.set_title("Independent test performance by event class")
    ax.legend(ncol=3, loc="upper center", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for containers in ax.containers:
        ax.bar_label(containers, fmt="%.2f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def choose_representatives(records):
    """Select figures automatically; user can replace names after visual inspection."""
    correct_cb = [r for r in records if {"connector", "break"}.issubset({e["type"] for e in r["gt"]})
                  and len(r["matches"]) >= 2]
    multiple = [r for r in records if len(r["gt"]) >= 3 and len(r["matches"]) >= 2]
    bend = [r for r in records if any(e["type"] == "bend" for e in r["gt"])
            and any(g["type"] == "bend" for g, _, _ in r["matches"])]
    error = sorted(records, key=lambda r: len(r["fn"]) + len(r["fp"]), reverse=True)
    return {
        "Figure_5a_correct_connector_break": correct_cb[0] if correct_cb else records[0],
        "Figure_5b_multiple_events": multiple[0] if multiple else records[min(1, len(records)-1)],
        "Figure_5c_bend_case": bend[0] if bend else records[min(2, len(records)-1)],
        "Figure_5d_error_case": error[0] if error else records[min(3, len(records)-1)],
    }


def main():
    ap = argparse.ArgumentParser(description="Create article-ready OTDR figures.")
    ap.add_argument("--data", required=True, help="test_dataset path")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--confidence", type=float, default=0.45)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    configure_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model)
    csvs = sorted(glob.glob(os.path.join(args.data, "*.csv")))

    records, match_rows, fn_rows, fp_rows = [], [], [], []
    t0 = time.perf_counter()

    # Prediction happens CSV-only. GT is read after predictions per trace only for plotting/evaluation.
    for n, csv_path in enumerate(csvs, 1):
        fname = os.path.basename(csv_path)
        try:
            km, db, pred = predict_csv_only(csv_path, bundle, args.confidence)
            gt = load_gt(os.path.splitext(csv_path)[0] + ".mask.json")
            matches, fn, fp = match(gt, pred, args.tolerance)
            record = {"file": fname, "csv_path": csv_path, "km": km, "db": db, "pred": pred,
                      "gt": gt, "matches": matches, "fn": fn, "fp": fp}
            records.append(record)
            for g, p, d in matches:
                match_rows.append({"file": fname, "gt_m": g["m"], "gt_type": g["type"],
                                   "pred_m": p["m"], "pred_type": p["type"],
                                   "confidence": p["confidence"], "distance_m": d})
            for g in fn:
                fn_rows.append({"file": fname, **g})
            for p in fp:
                fp_rows.append({"file": fname, **p})
            print(f"[{n:02d}/{len(csvs)}] {fname}: pred={len(pred)}, gt={len(gt)}, TP={len(matches)}")
        except Exception as e:
            print(f"[error] {fname}: {e}")

    matches_df = pd.DataFrame(match_rows)
    fn_df = pd.DataFrame(fn_rows)
    fp_df = pd.DataFrame(fp_rows)
    metrics = []
    for cls in ["bend", "connector", "break"]:
        tp = int((matches_df["gt_type"] == cls).sum()) if not matches_df.empty else 0
        fn = int((fn_df["type"] == cls).sum()) if not fn_df.empty else 0
        fp = int((fp_df["type"] == cls).sum()) if not fp_df.empty else 0
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*p*r/(p+r) if p+r else 0.0
        metrics.append({"class": cls, "TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r, "f1": f1})
    metrics_df = pd.DataFrame(metrics)

    # Figures and a journal-ready metric table.
    plot_confusion(matches_df, fn_df, fp_df, out / "Figure_4_test_event_matrix.png", args.dpi)
    plot_metrics(metrics_df, out / "Figure_6_per_class_performance.png", args.dpi)
    metrics_df.to_csv(out / "Table_4_test_metrics.csv", index=False)

    selected = choose_representatives(records)
    manifest = []
    for fig_name, rec in selected.items():
        out_path = out / f"{fig_name}.png"
        title = f"{fig_name.replace('_', ' ')}: {rec['file']}"
        plot_trace(rec["km"], rec["db"], rec["pred"], rec["gt"], title, out_path, args.dpi)
        manifest.append(f"{fig_name}.png | source={rec['file']} | GT={len(rec['gt'])} | "
                        f"TP={len(rec['matches'])} | FN={len(rec['fn'])} | FP={len(rec['fp'])}")

    manifest.extend([
        "Figure_4_test_event_matrix.png | event-level matching matrix on independent test dataset",
        "Figure_6_per_class_performance.png | precision, recall, and F1 by class",
        "Table_4_test_metrics.csv | numerical values used in Table 4",
    ])
    (out / "figure_manifest.txt").write_text("\n".join(manifest), encoding="utf-8")
    print(f"\nGenerated figures in {out} | elapsed={time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()
