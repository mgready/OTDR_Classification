"""
visualize_4class_test_predictions.py — PNG-визуализация первых N рефлектограмм
из test_dataset: prediction 4-class модели vs ручной Ground Truth.

ВАЖНО:
  1. Prediction строится ТОЛЬКО из CSV.
  2. JSON маска открывается только ПОСЛЕ prediction, исключительно для
     отрисовки/сравнения GT. В модель маска не подаётся.

На каждом PNG:
  - raw trace = светло-синяя линия
  - smoothed trace = тёмно-синяя линия
  - PRED = сплошные вертикальные линии с подписью event, metre, probability
  - GT = пунктирные вертикальные линии с подписью ручного event и metre
  - MATCH = зелёная полупрозрачная область, если PRED и GT одного типа
    совпали в пределах --tolerance (по умолчанию ±5m)

По умолчанию строит первые 10 CSV в алфавитном порядке.

Запуск:
  python visualize_4class_test_predictions.py --data "C:\\...\\test_dataset"

Например 10 файлов начиная с 20-го:
  python visualize_4class_test_predictions.py --data "C:\\...\\test_dataset" --start 20 --max-files 10

PNG сохраняются в:
  gt_event_model_data_4class/test_visualizations_4class/
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
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))
from otdr_common import parse_otdr_csv, trim_dead_zone

DEFAULT_MODEL = APP_DIR / "gt_event_model_data_4class" / "model_results" / "best_4class_event_classifier.joblib"
DEFAULT_OUT = APP_DIR / "gt_event_model_data_4class" / "test_visualizations_4class"
EVENT_CLASSES = {"bend", "connector", "break"}

PRED_COLORS = {"bend": "#E65100", "connector": "#00796B", "break": "#C2185B"}
GT_COLORS = {"bend": "#FFB300", "connector": "#43A047", "break": "#7B1FA2"}


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


def rolling_mean_std(x, radius):
    n = len(x)
    left = np.maximum(0, np.arange(n) - radius)
    right = np.minimum(n, np.arange(n) + radius + 1)
    cs = np.r_[0.0, np.cumsum(x, dtype=float)]
    cs2 = np.r_[0.0, np.cumsum(x * x, dtype=float)]
    cnt = (right - left).astype(float)
    mean = (cs[right] - cs[left]) / cnt
    var = (cs2[right] - cs2[left]) / cnt - mean * mean
    return mean, np.sqrt(np.maximum(var, 0.0))


def interval_mean_std(x, left, right):
    n = len(x)
    left = np.clip(np.asarray(left, int), 0, n)
    right = np.clip(np.asarray(right, int), 0, n)
    cnt = np.maximum(right - left, 1).astype(float)
    cs = np.r_[0.0, np.cumsum(x, dtype=float)]
    cs2 = np.r_[0.0, np.cumsum(x * x, dtype=float)]
    mean = (cs[right] - cs[left]) / cnt
    var = (cs2[right] - cs2[left]) / cnt - mean * mean
    return mean, np.sqrt(np.maximum(var, 0.0))


def local_slopes(km, db, idxs, width):
    pre = np.zeros(len(idxs)); post = np.zeros(len(idxs)); n = len(db)
    for j, idx in enumerate(idxs):
        a, b = max(0, idx-width), idx
        if b-a >= 3:
            x, y = km[a:b], db[a:b]
            xc = x-x.mean(); den = np.dot(xc, xc)
            pre[j] = np.dot(xc, y-y.mean()) / den if den > 0 else 0.0
        a, b = idx+1, min(n, idx+1+width)
        if b-a >= 3:
            x, y = km[a:b], db[a:b]
            xc = x-x.mean(); den = np.dot(xc, xc)
            post[j] = np.dot(xc, y-y.mean()) / den if den > 0 else 0.0
    return pre, post


def candidates(km, db):
    n = len(db)
    sm = moving_average(db, m_to_samples(km, 1.0))
    distance = m_to_samples(km, 1.5)
    mad = np.median(np.abs(sm - np.median(sm))) + 1e-6
    peaks, _ = find_peaks(sm, prominence=max(0.7, 1.5*mad), distance=distance)

    local_mean, _ = rolling_mean_std(sm, distance)
    step = local_mean - np.roll(local_mean, -distance)
    step[-distance:] = 0.0
    thr = max(0.25, float(np.percentile(step[distance:n-distance], 85))) if n > 2*distance else 0.25
    down, _ = find_peaks(step, height=thr, distance=distance)
    mins, _ = find_peaks(-sm, prominence=max(0.25, 0.5*mad), distance=distance)
    guard = m_to_samples(km, 5.0)
    idx = np.unique(np.concatenate([peaks, down, mins])).astype(int)
    return idx[(idx >= guard) & (idx < n-guard)]


def build_X(km, db, idxs, feature_names, expected_m=None):
    idxs = np.asarray(idxs, int); n = len(db)
    if len(idxs) == 0:
        return pd.DataFrame(columns=feature_names), {}
    ws, wc, wsl = m_to_samples(km, 1.5), m_to_samples(km, 6.0), m_to_samples(km, 5.0)
    grad = np.gradient(db, km)
    local_mean, local_std = rolling_mean_std(db, ws)
    pre_mean, pre_std = interval_mean_std(db, np.maximum(0,idxs-wc), idxs)
    post_mean, post_std = interval_mean_std(db, idxs+1, np.minimum(n,idxs+1+wc))
    pre_slope, post_slope = local_slopes(km, db, idxs, wsl)

    local_peak = np.array([np.max(db[max(0,i-ws):min(n,i+ws+1)]) for i in idxs])
    bg = (pre_mean+post_mean)/2
    peak_height = local_peak-bg
    max_pre_grad = np.array([np.max(grad[max(0,i-ws):i+1]) for i in idxs])
    min_post_grad = np.array([np.min(grad[i:min(n,i+ws+1)]) for i in idxs])
    local_range = np.array([np.ptp(db[max(0,i-ws):min(n,i+ws+1)]) for i in idxs])
    nominal = float(expected_m) if expected_m and expected_m > 0 else float(km[-1]*1000)

    values = {
        "m_norm": km[idxs]*1000/nominal,
        "db_at_event": db[idxs], "local_mean_db": local_mean[idxs], "local_std_db": local_std[idxs],
        "pre_mean_db": pre_mean, "post_mean_db": post_mean, "loss_dB": pre_mean-post_mean,
        "peak_above_bg_dB": peak_height, "pre_slope_dB_per_km": pre_slope,
        "post_slope_dB_per_km": post_slope, "slope_change_dB_per_km": post_slope-pre_slope,
        "derivative_at_event_dB_per_km": grad[idxs], "max_pre_derivative": max_pre_grad,
        "min_post_derivative": min_post_grad, "pre_std_db": pre_std, "post_std_db": post_std,
        "post_to_pre_std_ratio": post_std/(pre_std+1e-6), "peak_width_m": np.zeros(len(idxs)),
        "local_range_db": local_range,
    }
    return pd.DataFrame(values).reindex(columns=feature_names, fill_value=0.0), {"idxs":idxs, "m":km[idxs]*1000}


def nms(events, radius=5.0):
    kept=[]
    for e in sorted(events, key=lambda x:x["confidence"], reverse=True):
        if all(abs(e["m"]-k["m"]) > radius for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda x:x["m"])


def predict_csv_only(km, db, bundle, confidence, expected_m=None):
    idxs = candidates(km, db)
    model, features = bundle["model"], bundle["features"]
    X, aux = build_X(km, db, idxs, features, expected_m)
    if X.empty:
        return [], 0
    pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = [str(x) for x in model.classes_]
    events=[]
    for j, p in enumerate(pred.astype(str)):
        if p not in EVENT_CLASSES:
            continue
        c = float(proba[j, classes.index(p)])
        if c >= confidence:
            events.append({"m":float(aux["m"][j]), "type":p, "confidence":c, "idx":int(aux["idxs"][j])})
    return nms(events), len(idxs)


def load_gt_after_prediction(mask_path):
    if not os.path.exists(mask_path):
        return []
    data=json.load(open(mask_path, encoding="utf-8"))
    return [{"m":float(a["m"]),"type":a["type"]} for a in data.get("annotations", [])
            if a.get("type") in EVENT_CLASSES]


def match(gt, pred, tolerance):
    pairs=[]
    for gi,g in enumerate(gt):
        for pi,p in enumerate(pred):
            if g["type"] == p["type"]:
                d=abs(g["m"]-p["m"])
                if d <= tolerance: pairs.append((d,gi,pi))
    pairs.sort(); ug=set(); up=set(); matched=[]
    for d,gi,pi in pairs:
        if gi not in ug and pi not in up:
            ug.add(gi); up.add(pi); matched.append((gi,pi,d))
    return matched, ug, up


def plot_trace(km, db, pred, gt, matched, fname, output, tolerance):
    m=km*1000
    sm=moving_average(db,m_to_samples(km,1.0))
    fig,ax=plt.subplots(figsize=(17,6.5))
    ax.plot(m,db,color="#A9C4DA",lw=0.65,alpha=0.8,label="raw")
    ax.plot(m,sm,color="#1F3A5F",lw=1.15,label="smoothed")

    # Green bands mark exact correct type+position matches.
    for gi,pi,d in matched:
        center=(gt[gi]["m"]+pred[pi]["m"])/2
        ax.axvspan(center-tolerance,center+tolerance,color="#43A047",alpha=0.10,zorder=0)

    # Place labels at alternating vertical positions to reduce overlap.
    for j,e in enumerate(pred):
        color=PRED_COLORS[e["type"]]
        ypos=0.99 if j%2==0 else 0.80
        ax.axvline(e["m"],color=color,lw=1.8,ls="-",zorder=4)
        ax.annotate(f"PRED {e['type']}\n{e['m']:.1f}m | {e['confidence']:.0%}",
                    xy=(e["m"],ypos),xycoords=("data","axes fraction"),ha="center",va="top",
                    fontsize=7.5,color=color,bbox=dict(boxstyle="round,pad=0.22",fc="white",ec=color,alpha=0.92))

    for j,e in enumerate(gt):
        color=GT_COLORS[e["type"]]
        ypos=0.02 if j%2==0 else 0.19
        ax.axvline(e["m"],color=color,lw=1.6,ls="--",zorder=3)
        ax.annotate(f"GT {e['type']}\n{e['m']:.1f}m",xy=(e["m"],ypos),xycoords=("data","axes fraction"),
                    ha="center",va="bottom",fontsize=7.5,color=color,
                    bbox=dict(boxstyle="round,pad=0.22",fc="white",ec=color,alpha=0.92))

    legend=[
        plt.Line2D([0],[0],color="#A9C4DA",lw=1,label="raw"),
        plt.Line2D([0],[0],color="#1F3A5F",lw=1.5,label="smoothed"),
        plt.Line2D([0],[0],color="black",lw=1.7,ls="-",label="prediction (CSV only)"),
        plt.Line2D([0],[0],color="black",lw=1.5,ls="--",label="GT (JSON, evaluation only)"),
        plt.Rectangle((0,0),1,1,color="#43A047",alpha=0.15,label=f"correct match ±{tolerance}m"),
    ]
    ax.legend(handles=legend,loc="lower left",fontsize=8)
    ax.set_title(f"{fname} | predicted={len(pred)} | GT={len(gt)} | correct matches={len(matched)}",fontsize=10)
    ax.set_xlabel("distance (m)");ax.set_ylabel("dB");ax.grid(alpha=0.3)
    fig.tight_layout();fig.savefig(output,dpi=160,bbox_inches="tight");plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description="Visual audit: 4-class model predictions vs GT.")
    ap.add_argument("--data",required=True)
    ap.add_argument("--model",default=str(DEFAULT_MODEL))
    ap.add_argument("--out",default=str(DEFAULT_OUT))
    ap.add_argument("--confidence",type=float,default=0.45)
    ap.add_argument("--tolerance",type=float,default=5.0)
    ap.add_argument("--expected-m",type=float,default=None)
    ap.add_argument("--max-files",type=int,default=10)
    ap.add_argument("--start",type=int,default=0,help="0-based starting index in alphabetically sorted CSVs")
    args=ap.parse_args()

    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    bundle=joblib.load(args.model)
    csvs=sorted(glob.glob(os.path.join(args.data,"*.csv")))
    csvs=csvs[args.start:args.start+args.max_files]
    print(f"Model: {bundle.get('model_name','unknown')} | plotting {len(csvs)} files | start={args.start}")

    summary=[]
    for k,path in enumerate(csvs,1):
        fname=os.path.basename(path);started=time.perf_counter()
        try:
            # Phase 1: CSV-only model prediction.
            km,db,_=parse_otdr_csv(path);km,db=trim_dead_zone(km,db)
            pred,nc=predict_csv_only(km,db,bundle,args.confidence,args.expected_m)

            # Phase 2: only now JSON is opened for GT visualization.
            gt=load_gt_after_prediction(os.path.splitext(path)[0]+".mask.json")
            matched,_,_=match(gt,pred,args.tolerance)
            png=out/(Path(fname).stem+"_audit.png")
            plot_trace(km,db,pred,gt,matched,fname,png,args.tolerance)

            summary.append({"file":fname,"candidates":nc,"predicted":len(pred),"gt":len(gt),
                            "matches":len(matched),"png":png.name})
            print(f"[{k}/{len(csvs)}] {fname} | cand={nc} pred={len(pred)} GT={len(gt)} match={len(matched)} | {time.perf_counter()-started:.2f}s")
        except Exception as e:
            print(f"[error] {fname}: {e}")

    pd.DataFrame(summary).to_csv(out/"visual_audit_summary.csv",index=False)
    print(f"\nSaved {len(summary)} visual audits to: {out}")


if __name__=="__main__":
    main()
