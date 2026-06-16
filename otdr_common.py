"""
otdr_common.py — shared primitives for the OTDR pipeline.

Imported by splitter.py, classifier.py, checker.py so that parsing, the model
architecture, and localization are defined ONCE and stay identical across
train and test.

Classes:  0 = normal (class1) | 1 = bend (class2) | 2 = break (class3)
"""

from __future__ import annotations
import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CLASS_NAMES    = ["normal", "bend", "break"]
CLASS_DIRNAMES = ["class1", "class2", "class3"]      # sub-folder per class label
CLASS_PREFIXES = {"class1": 0, "class2": 1, "class3": 2}   # for raw filename labelling

N_POINTS = 512          # resampled sequence length (model input length)
DEAD_DB  = -40.0        # head samples below this are treated as launch dead-zone
SEED     = 42
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------- #
# PARSING  (semicolon-separated, comma decimal)
# --------------------------------------------------------------------------- #
def parse_otdr_csv(path):
    """Robust parse: any row whose first two ';'-separated fields are numbers (European
    comma-decimal) is treated as data; everything before the first such row is metadata.
    Tolerates header variations (missing/renamed 'km;dB' line, extra preamble)."""
    meta = {}
    dist, db = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines:
        p = ln.split(";")
        if len(p) >= 2:
            try:
                d = float(p[0].replace(",", ".").strip())
                v = float(p[1].replace(",", ".").strip())
                dist.append(d); db.append(v)
                continue
            except ValueError:
                pass
        if not dist and len(p) >= 2:               # still in the header block
            meta[p[0].strip()] = p[1].strip().replace(",", ".")
    if not dist:
        raise ValueError(f"No numeric data rows found in {path}")
    return np.asarray(dist, np.float64), np.asarray(db, np.float64), meta


def trim_dead_zone(dist, db, floor=DEAD_DB):
    k = 0
    while k < len(db) - 1 and db[k] < floor:
        k += 1
    return dist[k:], db[k:]


def label_from_filename(path):
    """Used by the splitter on the original flat dataset (class1_*.csv)."""
    name = os.path.basename(path).lower()
    return next((lab for pref, lab in CLASS_PREFIXES.items() if name.startswith(pref)), None)


# --------------------------------------------------------------------------- #
# RESAMPLING + CHANNELS
# --------------------------------------------------------------------------- #
def resample(dist_km, db, grid_km):
    return np.interp(grid_km, dist_km, db)


def make_channels(db_on_grid):
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    return np.stack([z(db_on_grid), z(np.gradient(db_on_grid))], axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# FOLDER LOADING  (reads split/<set>/<classN>/*.csv)
# --------------------------------------------------------------------------- #
def load_raw_folder(split_dir):
    """Return list of (dist, db, label, path) from a train/val/test folder."""
    raw = []
    for label, dirn in enumerate(CLASS_DIRNAMES):
        for fp in glob.glob(os.path.join(split_dir, dirn, "*.csv")):
            try:
                dist, db, _ = parse_otdr_csv(fp)
                dist, db = trim_dead_zone(dist, db)
                if len(dist) >= 8:
                    raw.append((dist, db, label, fp))
            except Exception as e:
                print(f"[skip] {os.path.basename(fp)}: {e}")
    return raw


def compute_grid(raw, n_points=N_POINTS):
    """Common physical grid from the TRAIN traces only (99th-pct window)."""
    max_km = float(np.percentile([d[-1] for d, *_ in raw], 99))
    return np.linspace(0.0, max_km, n_points)


def traces_to_X(raw, grid_km):
    X = np.stack([make_channels(resample(d, v, grid_km)) for d, v, *_ in raw])
    y = np.asarray([lab for *_, lab, _ in [(d, v, lab, p) for d, v, lab, p in raw]], np.int64)
    paths = [p for *_, p in raw]
    return X, y, paths


class OTDRDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i):  return self.X[i], self.y[i]


# --------------------------------------------------------------------------- #
# MODEL  (GAP + Linear -> CAM recoverable)
# --------------------------------------------------------------------------- #
class OTDRNet(nn.Module):
    def __init__(self, n_classes=3, in_ch=2, width=32):
        super().__init__()
        def block(i, o, k=7):
            p = k // 2
            return nn.Sequential(
                nn.Conv1d(i, o, k, padding=p), nn.BatchNorm1d(o), nn.ReLU(True),
                nn.Conv1d(o, o, k, padding=p), nn.BatchNorm1d(o), nn.ReLU(True),
                nn.MaxPool1d(2))
        self.features = nn.Sequential(
            block(in_ch, width), block(width, width * 2),
            block(width * 2, width * 4), block(width * 4, width * 4))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(width * 4, n_classes)

    def forward(self, x, return_features=False):
        feat = self.features(x)
        out = self.fc(self.gap(feat).squeeze(-1))
        return (out, feat) if return_features else out


# --------------------------------------------------------------------------- #
# CAM LOCALIZATION
# --------------------------------------------------------------------------- #
def compute_cam(model, x_2L, target_class):
    model.eval()
    xt = torch.from_numpy(x_2L[None]).to(DEVICE)
    with torch.no_grad():
        _, feat = model(xt, return_features=True)
        w = model.fc.weight[target_class].to(feat.device)   # Parameter: needs grad context off
        cam = torch.relu(torch.einsum("c,bcl->bl", w, feat)[0])
        cam = nn.functional.interpolate(cam[None, None], size=x_2L.shape[1],
                                        mode="linear", align_corners=False)[0, 0]
    cam = cam.cpu().numpy()
    return (cam - cam.min()) / (np.ptp(cam) + 1e-8)


def _smooth(x, w=5):
    if w < 2 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def find_noise_onset(db, win=6, std_thr=3.0, tail_frac_thr=0.6):
    """Index where the trailing noise floor begins (end of coherent backscatter).

    Valid signal is locally smooth (rolling std ~ 0-2 dB); the post-event noise floor
    swings wildly (rolling std >> that). Returns the earliest index from which the rest
    of the trace is predominantly noise. If the signal stays coherent to the end
    (a normal cable whose end sits inside the window), returns the last index."""
    n = len(db)
    rs = np.array([db[max(0, i - win):min(n, i + win + 1)].std() for i in range(n)])
    noisy = (rs > std_thr).astype(float)
    tail_frac = np.cumsum(noisy[::-1])[::-1] / np.arange(n, 0, -1)
    cand = np.where(tail_frac > tail_frac_thr)[0]
    return int(cand[0]) if cand.size else n - 1


def find_plateau_start(db, win=8, slope_thr=0.3):
    """First index after the launch pulse where the trace flattens into backscatter.
    Everything before this is launch settling, not a real fibre event."""
    n = len(db)
    sm = _smooth(db)
    peak = int(np.argmax(sm[:max(1, n // 5)]))     # launch peak (first ~20%)
    grad = np.abs(np.gradient(sm))
    for i in range(peak + 1, n - win):
        if np.all(grad[i:i + win] < slope_thr):    # sustained flat slope
            return i
    return min(peak + win, n - 1)


def localize_bend(db, grid_km, eof_idx, L=24, reflect_thr=3.0, min_drop=0.4):
    """Bend = a PERSISTENT non-reflective step-down in the otherwise-flat backscatter.

    Long median windows mean a transient dip (down then back up) scores ~0, while a
    true step (level drops and STAYS lower) scores its full magnitude. Launch tail is
    excluded by starting at the plateau; the fibre-end reflection is excluded by the
    reflective-spike test."""
    n = len(db)
    sm = _smooth(db)
    head = find_plateau_start(db)
    lo, hi = head + L, max(head + L + 1, eof_idx - L)
    best_i, best_drop = None, -1e9
    for i in range(lo, hi):
        pre, post = np.median(sm[i - L:i]), np.median(sm[i:i + L])
        drop = pre - post                                  # persistent level loss
        local_peak = sm[i - L:i + L].max() - max(pre, post)
        if local_peak > reflect_thr:                       # reflective -> not a bend
            continue
        if drop > best_drop:
            best_drop, best_i = drop, i
    if best_i is None or best_drop < min_drop:             # no clear step: steepest slope
        seg = np.gradient(sm)[lo:hi] if hi > lo else np.array([0.0])
        best_i = lo + int(np.argmin(seg))
    to_m = lambda j: float(grid_km[j] * 1000.0)
    half = max(L // 2, 5)
    return to_m(best_i), (to_m(max(0, best_i - half)), to_m(min(n - 1, best_i + half)))


def localize_break(db, grid_km, eof_idx, win=8, reflect_thr=3.0):
    """Break = the FIRST prominent Fresnel reflection after the launch.

    Handles both cases: a full break (reflection right before the signal dies into
    noise) and a partial / near break (reflection mid-fibre, signal continues at a
    lower level). The always-present fibre-end reflection is never the first peak when
    a real break exists earlier, so taking the first qualifying peak is correct."""
    n = len(db)
    sm = _smooth(db)
    head = find_plateau_start(db)
    last = min(eof_idx + int(0.04 * n), n - win - 1)
    for i in range(head + win, last):
        left = np.median(sm[i - win:i])
        rise = sm[i] - left
        if rise > reflect_thr and sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]:
            j = i
            while j > head and sm[j - 1] < sm[j]:          # walk back to foot of rise
                j -= 1
            to_m = lambda k: float(grid_km[k] * 1000.0)
            return to_m(j), (to_m(j), to_m(min(n - 1, i + win)))
    to_m = lambda k: float(grid_km[k] * 1000.0)            # fallback: noise onset
    return to_m(eof_idx), (to_m(max(0, eof_idx - win)), to_m(min(n - 1, eof_idx)))


def predict_file(path, model, grid_km):
    dist, db, _ = parse_otdr_csv(path)
    dist, db = trim_dead_zone(dist, db)
    db_grid = resample(dist, db, grid_km)
    x = make_channels(db_grid)
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(torch.from_numpy(x[None]).to(DEVICE)), 1)[0].cpu().numpy()
    cls = int(prob.argmax())
    r = {"file": os.path.basename(path), "class": cls, "class_name": CLASS_NAMES[cls],
         "confidence": float(prob[cls]),
         "prob": {CLASS_NAMES[i]: float(prob[i]) for i in range(3)},
         "db_grid": db_grid, "grid_m": grid_km * 1000.0,
         "event_m": None, "zone_m": None, "cam": None}
    if cls == 1:        # bend -> non-reflective step-down
        eof = find_noise_onset(db_grid)
        ev, zone = localize_bend(db_grid, grid_km, eof)
        r.update(event_m=ev, zone_m=zone)
    elif cls == 2:      # break -> rising edge of the reflective peak before the noise
        eof = find_noise_onset(db_grid)
        ev, zone = localize_break(db_grid, grid_km, eof)
        r.update(event_m=ev, zone_m=zone)
    return r


def plot_localization(result, save=None):
    import matplotlib.pyplot as plt
    g, db = result["grid_m"], result["db_grid"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(g, db, lw=1.3, color="#1f3a5f")
    title = f"{result['file']} -> {result['class_name'].upper()} ({result['confidence']:.0%})"
    if result["event_m"] is not None:
        lo, hi = result["zone_m"]
        ax.axvspan(lo, hi, color="orange", alpha=0.25, label="event zone")
        ax.axvline(result["event_m"], color="red", lw=1.6, ls="--",
                   label=f"event @ {result['event_m']:.1f} m")
        ax.legend(loc="lower left")
        title += f"  |  event @ {result['event_m']:.1f} m"
    ax.set_xlabel("distance (m)"); ax.set_ylabel("dB"); ax.set_title(title)
    ax.grid(alpha=0.3); fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# CHECKPOINT
# --------------------------------------------------------------------------- #
def save_checkpoint(model, grid_km, path):
    torch.save({"state_dict": model.state_dict(), "grid_km": grid_km,
                "class_names": CLASS_NAMES}, path)
    print("saved", path)


def load_checkpoint(path):
    # weights_only=False: the checkpoint holds the numpy resampling grid, and it is
    # your own trusted file. PyTorch 2.6 defaults this to True and would refuse it.
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = OTDRNet().to(DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["grid_km"]