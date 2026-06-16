"""
otdr_channel.py — regression-channel trend analysis -> features -> 3-class + event.

The pipeline your supervisor sketched:

    1. Fit ONE central trend (robust regression) over the coherent backscatter.
    2. Build a parallel channel (trend +/- k * robust-noise) -> the "optimal channel".
    3. Mark local maxima (green) / minima (red); points outside the channel are
       candidate anomalies.
    4. Turn the trend behaviour into FEATURES (slope, channel width, where the signal
       leaves the channel, how far the trend reaches vs the expected length).
    5. Classify  ->  class 1 normal | class 2 bend | class 3 break,  and locate the event.

Everything is in physical units (m, dB, dB/km, fractions), so the same code and the
same thresholds apply to a 150 m lab fibre or a 5 km field span.

The class is currently decided by transparent RULES on the features. The same feature
vector (see trend_features) can instead be fed to a tiny trained classifier later --
that is the "learned weights on trend features" version, and it stays length-independent.

Depends on: numpy, scipy, matplotlib + parse_otdr_csv from otdr_common,
            find_valid_region from otdr_trends.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import theilslopes
from otdr_common import parse_otdr_csv, trim_dead_zone
from otdr_trends import find_valid_region, _smooth, _m_to_samples


# --------------------------------------------------------------------------- #
def mad_std(x):
    """Robust std via Median Absolute Deviation — ignores spikes/events."""
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + 1e-9


def fit_channel(km, db, start, eof, sigma_mult=2.5):
    """Central trend (robust) + parallel channel + extrema + anomalies, over [start, eof]."""
    x, y = km[start:eof], _smooth(db, _m_to_samples(km, 1.0))[start:eof]

    # robust central trend (Theil-Sen ignores the bend step / outliers)
    slope, intercept, _, _ = theilslopes(y, x)
    trend = slope * x + intercept
    resid = y - trend

    noise = mad_std(resid)
    thr = sigma_mult * noise
    upper, lower = trend + thr, trend - thr

    anom_above = y > upper
    anom_below = y < lower

    # local extrema for the green/red dots
    gmax, _ = find_peaks(y, distance=_m_to_samples(km, 2.0))
    rmin, _ = find_peaks(-y, distance=_m_to_samples(km, 2.0))

    return {"x": x, "y": y, "trend": trend, "resid": resid,
            "upper": upper, "lower": lower, "slope": float(slope),
            "intercept": float(intercept), "thr": float(thr), "noise": float(noise),
            "anom_above": anom_above, "anom_below": anom_below,
            "gmax": gmax, "rmin": rmin}


# --------------------------------------------------------------------------- #
def _sustained_step(y, win):
    """Largest sustained pre/post median DROP and its local index within y."""
    n = len(y)
    best_i, best = 0, -1e9
    for i in range(win, n - win):
        d = np.median(y[i - win:i]) - np.median(y[i:i + win])
        if d > best:
            best, best_i = d, i
    return best_i, float(best)


def trend_features(km, db, start, eof, ch, expected_length_m):
    """Derive the feature vector the classifier reasons over."""
    x = ch["x"]
    valid_len_m = float((km[eof] - km[start]) * 1000.0)
    valid_frac = valid_len_m / expected_length_m if expected_length_m else 1.0

    win = _m_to_samples(km, 3.0)
    step_i, step_drop = _sustained_step(ch["y"], win)          # bend signature
    step_m = float(x[step_i] * 1000.0)

    # reflective spike just before the trend terminates (break signature)
    tail = ch["y"][max(0, len(ch["y"]) - _m_to_samples(km, 8.0)):]
    end_reflect = float(tail.max() - np.median(ch["y"]))

    return {
        "slope_dB_per_km": ch["slope"],
        "channel_halfwidth_dB": ch["thr"],
        "noise_dB": ch["noise"],
        "valid_length_m": round(valid_len_m, 1),
        "valid_fraction": round(valid_frac, 3),
        "step_drop_dB": round(step_drop, 2),
        "step_location_m": round(step_m, 1),
        "end_reflect_dB": round(end_reflect, 2),
        "n_anom_below": int(ch["anom_below"].sum()),
        "n_anom_above": int(ch["anom_above"].sum()),
    }


# --------------------------------------------------------------------------- #
def classify_from_trend(feat, eof_m,
                        break_frac=0.85, bend_drop_db=0.8, reflect_db=3.0):
    """Rule-based 3-class decision + event location, from trend features."""
    # class 3: the trend dies well before the fibre should end (with a reflection)
    if feat["valid_fraction"] < break_frac:
        return {"class": 3, "class_name": "break",
                "event_m": round(eof_m, 1),
                "why": f"trend ends at {feat['valid_length_m']} m "
                       f"({feat['valid_fraction']*100:.0f}% of expected)"}
    # class 2: a sustained step-down below the channel mid-fibre
    if feat["step_drop_dB"] > bend_drop_db:
        return {"class": 2, "class_name": "bend",
                "event_m": feat["step_location_m"],
                "why": f"sustained {feat['step_drop_dB']:.1f} dB step at "
                       f"{feat['step_location_m']} m"}
    # otherwise: clean trend to the end
    return {"class": 1, "class_name": "normal", "event_m": None,
            "why": "signal stays within channel, trend reaches expected length"}


# --------------------------------------------------------------------------- #
def analyze_file(path, expected_length_m=150.0, sigma_mult=2.5):
    dist, db, meta = parse_otdr_csv(path)
    dist, db = trim_dead_zone(dist, db)
    start, eof = find_valid_region(dist, db)
    ch = fit_channel(dist, db, start, eof, sigma_mult=sigma_mult)
    feat = trend_features(dist, db, start, eof, ch, expected_length_m)
    eof_m = float(dist[eof] * 1000.0)
    result = classify_from_trend(feat, eof_m)
    return {"file": path, "km": dist, "db": db, "start": start, "eof": eof,
            "channel": ch, "features": feat, "result": result, "meta": meta}


# --------------------------------------------------------------------------- #
def plot_channel(out, save=None, full=True):
    """Full OTDR trace with the trend + parallel channel overlaid on the valid region,
    plus the class verdict and event marker. Set full=False to crop to the channel."""
    import matplotlib.pyplot as plt
    km, db = out["km"], out["db"]
    start, eof = out["start"], out["eof"]
    ch, r = out["channel"], out["result"]
    x = ch["x"]
    fig, ax = plt.subplots(figsize=(13, 5))

    if full:
        ax.plot(km, db, lw=0.7, color="#9DB8D2", alpha=0.7, label="full OTDR (raw)")
        if start > 0:
            ax.axvspan(km[0], km[start], color="purple", alpha=0.08, label="launch / dead-zone")
        if eof < len(km) - 1:
            ax.axvspan(km[eof], km[-1], color="brown", alpha=0.08, label="noise floor")
    else:
        ax.plot(x, ch["y"], lw=0.9, color="#6FA8DC", alpha=0.8, label="RAW signal")

    ax.plot(x, ch["trend"], lw=2.6, ls="--", color="#D32F2F",
            label=f"Central trend (slope={ch['slope']:.4f})")
    ax.plot(x, ch["upper"], lw=2.0, color="#E69138", label="Upper channel")
    ax.plot(x, ch["lower"], lw=2.0, color="#2E7D32", label="Lower channel")
    ax.scatter(x[ch["gmax"]], ch["y"][ch["gmax"]], s=24, color="#2E7D32", zorder=5)
    ax.scatter(x[ch["rmin"]], ch["y"][ch["rmin"]], s=24, color="#CC0000", zorder=5)

    if r["event_m"] is not None:
        ax.axvline(r["event_m"] / 1000.0, color="black", ls=":", lw=1.8,
                   label=f"{r['class_name']} @ {r['event_m']:.1f} m")

    ax.set_title(f"OTDR  ->  CLASS {r['class']}: {r['class_name'].upper()}   ({r['why']})")
    ax.set_xlabel("distance (km)"); ax.set_ylabel("dB")
    ax.grid(alpha=0.35); ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\PCA\Desktop\OTDR_CLassification\Dataset\class1_2026_04_27_14_37_00.csv"
    out = analyze_file(path, expected_length_m=150.0)
    print(f"\n{out['file']}")
    print("trend features:")
    for k, v in out["features"].items():
        print(f"   {k:22s}: {v}")
    print(f"\nverdict: CLASS {out['result']['class']} "
          f"({out['result']['class_name']}) -- {out['result']['why']}")
    if out["result"]["event_m"] is not None:
        print(f"event at {out['result']['event_m']} m")
    plot_channel(out, save="channel.png")
    print("\nsaved channel.png")