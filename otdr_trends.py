"""
otdr_trends.py — piecewise-linear trend decomposition + event extraction.

This is the "trend-based" replacement for whole-file classification. It works on
the RAW (distance, dB) trace at its native resolution, using distance-based windows,
so it is length-independent: the same code handles a 150 m lab fibre or a 5 km field
span. No fixed-length resampling, no trained model required.

Output per trace:
    segments : list of straight trend lines  {km0, km1, slope_dB_per_km, intercept}
    events   : list of {km, type, loss_dB, reflectance_dB}
               type in {bend, connector, break, end}

A bend  = non-reflective step-down (loss, no reflection).
A break = reflective spike followed by collapse to the noise floor.
A connector/reflective event = reflection where the signal continues.
The "end" = the terminal reflection at (near) the expected fibre length.

These segment slopes/intercepts are the "trend weights" your supervisor described;
the events fall out of the discontinuities between them. If you later train a model
to PREDICT these, the dicts below are exactly the labels to regress against — you
correct a handful by hand instead of drawing all of them.

Depends on: numpy, scipy, matplotlib, and parse_otdr_csv from otdr_common.
"""

import numpy as np
from scipy.signal import find_peaks
from otdr_common import parse_otdr_csv, trim_dead_zone


# --------------------------------------------------------------------------- #
def _smooth(y, w):
    if w < 3 or len(y) < w:
        return y.astype(float)
    w = int(w) | 1                      # force odd
    return np.convolve(y, np.ones(w) / w, mode="same")


def _m_to_samples(km, meters):
    """Convert a window in metres to a sample count at this trace's resolution."""
    dx_km = float(np.median(np.diff(km)))           # km per sample
    return max(2, int((meters / 1000.0) / dx_km))


# --------------------------------------------------------------------------- #
def find_launch_start(km, db, w, slope_thr_db_per_km=60.0):
    """First flat-slope index after the launch peak (end of the dead-zone)."""
    n = len(db)
    sm = _smooth(db, w)
    peak = int(np.argmax(sm[:max(1, n // 5)]))
    grad_km = np.abs(np.gradient(sm, km))
    for i in range(peak + 1, max(peak + 2, n - w)):
        if np.all(grad_km[i:i + w] < slope_thr_db_per_km):
            return i
    return min(peak + w, n - 1)


def _final_run_eof(mask, w, start):
    """eof = start of the final density-smoothed True run that reaches the window end."""
    n = len(mask)
    if not mask.any():
        return n - 1
    dens = _smooth(mask.astype(float), 2 * w) > 0.5
    if dens[-1]:
        i = n - 1
        while i > start and dens[i - 1]:
            i -= 1
        return i
    idx = np.where(dens)[0]
    return int(idx[-1]) if idx.size else n - 1


def _noise_variance(km, db, w, start, std_thr=3.0, **_):
    """Noise = sustained high local variance (floor at -50 OR oscillating high)."""
    n = len(db)
    rs = np.array([db[max(0, i - w):min(n, i + w + 1)].std() for i in range(n)])
    return _final_run_eof(rs > std_thr, w, start)


def _noise_level(km, db, w, start, drop_margin_db=8.0, level_win_m=12.0, **_):
    """Noise = signal far below the backscatter reference level."""
    n = len(db)
    sm = _smooth(db, w)
    lw = _m_to_samples(km, level_win_m)
    level_ref = float(np.median(sm[start:min(n, start + lw)]))
    return _final_run_eof(sm < level_ref - drop_margin_db, w, start)


def _noise_gradient(km, db, w, start, **_):
    """Noise begins at the bottom of the steepest sustained collapse after backscatter."""
    n = len(db)
    sm = _smooth(db, w)
    g = np.gradient(sm, km)
    gi = start + int(np.argmin(g[start:n]))        # steepest drop (the collapse)
    i = gi
    while i < n - 1 and sm[i + 1] <= sm[i]:         # walk to the bottom of the drop
        i += 1
    return i


REGION_METHODS = {"variance": _noise_variance, "level": _noise_level, "gradient": _noise_gradient}


def find_valid_region(km, db, method="variance", noise_win_m=2.5,
                      slope_thr_db_per_km=60.0, launch_guard_m=5.0,
                      reflect_thr_db=3.0, max_extend_passes=3, **kw):
    """
    (start, eof) с итеративным расширением: если сразу после найденного eof
    стоит острый рефлективный пик (connector), и после него на протяжении
    хотя бы noise_win_m*4 метров сигнал ОСТАЁТСЯ стабильным (не хаотичным) —
    значит это был промежуточный connector, а не настоящий конец волокна.
    В этом случае продолжаем искать eof дальше, начиная от пика.
    """
    from scipy.signal import find_peaks

    w = _m_to_samples(km, noise_win_m)
    start = find_launch_start(km, db, w, slope_thr_db_per_km)
    guard_samples = _m_to_samples(km, launch_guard_m)
    start = min(start + guard_samples, len(db) - 2)

    fn = REGION_METHODS.get(method, _noise_variance)
    search_start = start
    eof = fn(km, db, w, search_start, **kw)

    check_w = _m_to_samples(km, noise_win_m * 4)
    for _ in range(max_extend_passes):
        # есть ли рефлективный пик ПРЯМО НА границе eof (в пределах noise_win_m)?
        window_lo, window_hi = max(0, eof - w), min(len(db), eof + w)
        sm_local = _smooth(db[max(0, eof - 3*w):min(len(db), eof + 3*w)], max(3, w // 2))
        local_peaks, _ = find_peaks(sm_local, prominence=reflect_thr_db)
        if len(local_peaks) == 0:
            break  # обычный шум, не промежуточный connector -> останавливаемся

        # проверяем: после этого пика сигнал СТАБИЛЕН ещё check_w самплов?
        after_peak = min(len(db), eof + w + check_w)
        if after_peak - (eof + w) < check_w * 0.5:
            break  # данных не хватает для проверки, останавливаемся на этом eof
        segment_after = db[eof + w: after_peak]
        if segment_after.std() > kw.get("std_thr", 3.0) * 1.5:
            break  # после пика всё равно хаос -> это был настоящий конец, не продолжаем

        # сигнал после пика стабилен -> это промежуточный connector, ищем eof ДАЛЬШЕ
        new_search_start = eof + w
        new_eof = fn(km, db, w, new_search_start, **kw)
        if new_eof <= eof:
            break
        eof = new_eof

    return start, max(start + 1, eof)


# --------------------------------------------------------------------------- #
def detect_breakpoints(km, db, start, eof, win_m=1.5,
                       loss_thr_db=0.5, reflect_thr_db=3.0, merge_m=2.0):
    """Find event indices in [start, eof): reflective peaks and non-reflective steps."""
    w = _m_to_samples(km, win_m)
    sm = _smooth(db, w)
    n = len(db)

    # reflective events: prominent positive peaks
    refl, _ = find_peaks(sm, prominence=reflect_thr_db)
    refl = [int(i) for i in refl if start < i < eof]

    # non-reflective loss steps: peaks of (pre-median minus post-median)
    step = np.zeros(n)
    for i in range(start + w, eof - w):
        step[i] = np.median(sm[i - w:i]) - np.median(sm[i:i + w])
    loss, _ = find_peaks(step, height=loss_thr_db, distance=w)
    merge_km = merge_m / 1000.0
    loss = [int(i) for i in loss
            if start < i < eof and all(abs(km[i] - km[r]) > merge_km for r in refl)]

    return sorted(refl + loss), set(refl), sm


# --------------------------------------------------------------------------- #
def fit_trends(km, db, breakpoints, start, eof, min_seg_m=2.0):
    """OLS straight-line fit on each segment between breakpoints."""
    w = _m_to_samples(km, min_seg_m)
    sm = _smooth(db, _m_to_samples(km, 1.0))
    bps = sorted(set([start] + [b for b in breakpoints if start < b < eof] + [eof]))
    segs = []
    for a, b in zip(bps[:-1], bps[1:]):
        if b - a < max(3, w):
            continue
        slope, intercept = np.polyfit(km[a:b], sm[a:b], 1)
        segs.append({"i0": a, "i1": b,
                     "km0": float(km[a]), "km1": float(km[b - 1]),
                     "slope_dB_per_km": float(slope), "intercept": float(intercept)})
    return segs


# --------------------------------------------------------------------------- #
def classify_events(km, db, breakpoints, refl_set, eof, sm,
                    win_m=1.5, die_within_m=3.0, expected_length_m=None):
    """Turn breakpoints into typed events with loss / reflectance."""
    w = _m_to_samples(km, win_m)
    n = len(db)
    die_km = die_within_m / 1000.0
    events = []
    for i in breakpoints:
        pre = np.median(sm[max(0, i - w):i])
        post = np.median(sm[i:min(n, i + w)])
        loss = float(pre - post)
        reflectance = float(sm[max(0, i - w):min(n, i + w)].max() - max(pre, post))
        is_refl = i in refl_set
        dies = (km[eof] - km[i]) < die_km            # noise begins just after this event

        if is_refl and dies:
            if expected_length_m is not None and km[i] * 1000.0 < 0.9 * expected_length_m:
                etype = "break"
            elif expected_length_m is not None:
                etype = "end"
            else:
                etype = "break/end"
        elif is_refl:
            etype = "connector"
        else:
            etype = "bend"
        events.append({"km": float(km[i]), "m": float(km[i] * 1000.0),
                       "type": etype, "loss_dB": round(loss, 2),
                       "reflectance_dB": round(reflectance, 2)})
    return events


# --------------------------------------------------------------------------- #
def merge_events(events, merge_m=12.0):
    """Collapse runs of the SAME-type events within merge_m metres into one event.
    A continuous bend region becomes a single bend (loss accumulated) instead of many."""
    if not events:
        return events
    ev = sorted(events, key=lambda e: e["m"])
    out = [dict(ev[0])]
    for e in ev[1:]:
        last = out[-1]
        if e["type"] == last["type"] and (e["m"] - last["m"]) <= merge_m:
            last["loss_dB"] = round(float(last.get("loss_dB", 0.0)) + float(e.get("loss_dB", 0.0)), 2)
            last["reflectance_dB"] = round(max(float(last.get("reflectance_dB", 0.0)),
                                               float(e.get("reflectance_dB", 0.0))), 2)
        else:
            out.append(dict(e))
    return out


def analyze_trends(km, db, expected_length_m=None, region_method="variance",
                   loss_thr_db=0.5, reflect_thr_db=3.0, win_m=1.5,
                   end_frac=0.90, merge_m=12.0, snap_end=True, snap_tol=0.12,
                   launch_guard_m=5.0):
    """Full decomposition of one (km, dB) trace.

    - merge_m : consecutive same-type events within this many metres collapse into one,
                so a continuous bend reads as ONE bend instead of 7-12.
    - terminal: находит ПОСЛЕДНИЙ рефлективный пик перед eof как terminus (без
                ограничения "только последние 20м" — это и было причиной путаницы
                connector@127m / break@143m). Дальше break/end решается по
                expected_length_m, но САМ ПОИСК terminus от него не зависит."""
    from otdr_trends import detect_breakpoints, fit_trends, classify_events, merge_events

    start, eof = find_valid_region(km, db, method=region_method, launch_guard_m=launch_guard_m)
    bps, refl_set, sm = detect_breakpoints(km, db, start, eof,
                                           win_m=win_m, loss_thr_db=loss_thr_db,
                                           reflect_thr_db=reflect_thr_db)
    segs = fit_trends(km, db, bps, start, eof)
    events = classify_events(km, db, bps, refl_set, eof, sm,
                             win_m=win_m, expected_length_m=expected_length_m)
    events = merge_events(events, merge_m=merge_m)

    # terminus = последний рефлективный пик перед началом шума (eof),
    # БЕЗ ограничения "только последние 20м" — это чинит баг с connector@127/break@143.
    if refl_set:
        term_idx = max(i for i in refl_set if i < eof)
    else:
        term_idx = eof

    term_m = float(km[term_idx] * 1000.0)
    events = [e for e in events if abs(e["m"] - term_m) > 5.0]

    if expected_length_m:
        if term_m < expected_length_m * end_frac:
            pos, ttype = term_m, "break"
        elif abs(term_m - expected_length_m) <= snap_tol * expected_length_m:
            pos, ttype = (expected_length_m if snap_end else term_m), "end"
        else:
            pos, ttype = term_m, "end"
    else:
        pos, ttype = term_m, "end"

    # ЭТО БЫЛО ПОТЕРЯНО — обязательно добавляем terminus-событие в список:
    events.append({"km": pos / 1000.0, "m": pos, "type": ttype,
                   "loss_dB": 0.0, "reflectance_dB": 0.0})

    # И ЭТО БЫЛО ПОТЕРЯНО — обязательный return, без него функция возвращала None:
    return {"km": km, "db": db, "smooth": sm,
            "start": start, "eof": eof,
            "segments": segs, "events": events}


def analyze_file(path, **kw):
    dist, db, meta = parse_otdr_csv(path)
    dist, db = trim_dead_zone(dist, db)
    out = analyze_trends(dist, db, **kw)
    out["meta"] = meta
    out["file"] = path
    return out


# --------------------------------------------------------------------------- #
def plot_trends(result, save=None):
    """Raw trace + per-segment trend lines + typed event markers."""
    import matplotlib.pyplot as plt
    km, db, sm = result["km"], result["db"], result["smooth"]
    m = km * 1000.0
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(m, db, lw=0.8, color="#A8C5E0", label="raw", zorder=1)
    ax.plot(m, sm, lw=1.3, color="#1f3a5f", label="smoothed", zorder=2)

    for s in result["segments"]:
        xs = km[s["i0"]:s["i1"]]
        ys = s["slope_dB_per_km"] * xs + s["intercept"]
        ax.plot(xs * 1000.0, ys, lw=2.2, color="#D32F2F", zorder=3)

    colors = {"bend": "#FF9800", "break": "#C2185B", "end": "#455A64",
              "connector": "#2E7D32", "break/end": "#C2185B"}
    for e in result["events"]:
        c = colors.get(e["type"], "#000000")
        ax.axvline(e["m"], color=c, ls="--", lw=1.5, zorder=4)
        ax.annotate(f"{e['type']}\n{e['m']:.1f} m\n{e['loss_dB']:+.1f} dB",
                    xy=(e["m"], ax.get_ylim()[1]), fontsize=8, color=c,
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.85))
    ax.axvline(m[result["eof"]], color="gray", ls=":", lw=1, label="noise onset")

    ax.set_xlabel("distance (m)"); ax.set_ylabel("dB")
    ax.set_title(f"{result.get('file','trace')}  |  "
                 f"{len(result['segments'])} trend segments, "
                 f"{len(result['events'])} events")
    ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=8); fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\PCA\Desktop\OTDR_CLassification\Dataset\class2_2026_04_27_15_25_54.csv"
    r = analyze_file(path, expected_length_m=150.0)
    print(f"\n{r['file']}")
    print("segments (trend weights):")
    for s in r["segments"]:
        print(f"   {s['km0']*1000:6.1f}-{s['km1']*1000:6.1f} m | "
              f"slope {s['slope_dB_per_km']:+7.2f} dB/km | intercept {s['intercept']:+6.2f}")
    print("events:")
    for e in r["events"]:
        print(f"   {e['m']:6.1f} m | {e['type']:10s} | loss {e['loss_dB']:+5.1f} dB | "
              f"refl {e['reflectance_dB']:+5.1f} dB")
    plot_trends(r, save="trends.png")
    print("\nsaved trends.png")