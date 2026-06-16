"""
detector_trained.py — use the trained trend classifier inside otdr_studio.

In the Studio, choose the detector dropdown -> "Load custom model…" -> pick THIS file.
The viewer will then overlay the piecewise trends and show the trained model's class
verdict (with probability) live, on whatever file you browse to.

Contract expected by otdr_studio:  detect(km, db, db_s, zones_m, params) -> result dict.
"""

import numpy as np
import joblib
from otdr_trends import analyze_trends
from otdr_cluster import trace_features

MODEL_PATH = r"C:\Users\PCA\Desktop\OTDR_CLassification\trend_clf.joblib"
_BUNDLE = None


def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE


def detect(km, db, db_s, zones_m, params):
    b = _bundle()
    exp = params.get("expected_m", b["expected_m"])

    out = analyze_trends(km, db, expected_length_m=exp,
                         region_method=params.get("region_method", "variance"))
    feat = trace_features(out, exp)
    x = np.array([[feat[c] for c in b["features"]]])
    cls = int(b["model"].predict(x)[0])
    proba = b["model"].predict_proba(x)[0]
    pmap = {b["names"][c]: float(p) for c, p in zip(b["model"].classes_, proba)}
    name = b["names"][cls]

    trends = []
    for s in out["segments"]:
        xs = km[s["i0"]:s["i1"]] * 1000.0
        ys = s["slope_dB_per_km"] * km[s["i0"]:s["i1"]] + s["intercept"]
        trends.append((xs, ys, "#D32F2F", f"{s['slope_dB_per_km']:.2f} dB/km"))

    feat_show = {k: round(feat[k], 3) for k in b["features"]}
    feat_show.update({f"p_{n}": round(p, 2) for n, p in pmap.items()})

    return {
        "trends": trends,
        "channel": None,
        "events": out["events"],
        "verdict": f"CLASS {cls}: {name.upper()}  ({pmap.get(name, 0.0)*100:.0f}%)",
        "region": (km[out["start"]] * 1000.0, km[out["eof"]] * 1000.0),
        "features": feat_show,
    }