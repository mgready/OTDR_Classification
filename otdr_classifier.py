"""
OTDR trace classification + weakly-supervised event localization.

Classes:
    0 = normal (default position)
    1 = bend   (macrobend / manipulation)
    2 = break  (disconnection)

Pipeline:
    parse CSV  ->  trim launch dead-zone  ->  resample to a fixed physical grid
    ->  build [dB, dDB/dx] channels  ->  1D CNN with a CAM head
    ->  predicted class + Class Activation Map
    ->  CAM peak refined with the derivative  ->  event zone in metres.

Why CAM:
    You have one CLASS label per file, not per-sample event positions. A CNN that
    ends in Global Average Pooling + a single Linear layer yields a Class Activation
    Map (Zhou et al., 2016) almost for free: a 1-D heat-map over the fibre telling
    you WHERE the discriminative event is. That is what lets you "highlight the zone"
    without ever labelling event locations by hand.

Expected folder layout: all CSVs in one folder, labelled by filename prefix
(edit CLASS_PREFIXES below if yours differ):

    Dataset/
        class1_*.csv   normal (default)
        class2_*.csv   bend
        class3_*.csv   break

Author scaffold for Magzhan — run section by section; it is written to be debugged
incrementally rather than as a black box.
"""

from __future__ import annotations
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------------------------------- #
# CONFIG  — edit these three paths to point at your class sub-folders.
# --------------------------------------------------------------------------- #
ROOT = r"C:\Users\PCA\Desktop\OTDR_CLassification\Dataset"
# Files live in one folder, labelled by filename prefix: class1_*.csv etc.
CLASS_PREFIXES = {
    "class1": 0,   # normal (default)
    "class2": 1,   # bend
    "class3": 2,   # break
}
CLASS_NAMES = ["normal", "bend", "break"]

CABLE_LENGTH_M = 150.0      # used only for plotting / sanity, grid is auto-detected
N_POINTS       = 512        # resampled sequence length (model input length)
DEAD_DB        = -40.0      # samples below this at the head are treated as dead-zone
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------- #
# 1. PARSING  — semicolon-separated, comma decimal (European format).
# --------------------------------------------------------------------------- #
def parse_otdr_csv(path):
    """Return (distance_km: np.ndarray, db: np.ndarray, meta: dict)."""
    meta = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    data_start = None
    for i, ln in enumerate(lines):
        if ln.lower().replace(" ", "").startswith("km;db"):
            data_start = i + 1
            for m in lines[:i]:                       # metadata above the km;dB row
                parts = m.split(";")
                if len(parts) >= 2:
                    meta[parts[0].strip()] = parts[1].strip().replace(",", ".")
            break
    if data_start is None:
        raise ValueError(f"No 'km;dB' header found in {path}")

    dist, db = [], []
    for ln in lines[data_start:]:
        parts = ln.split(";")
        if len(parts) < 2:
            continue
        try:
            d = float(parts[0].replace(",", "."))
            v = float(parts[1].replace(",", "."))
        except ValueError:
            continue
        dist.append(d)
        db.append(v)
    return np.asarray(dist, dtype=np.float64), np.asarray(db, dtype=np.float64), meta


def trim_dead_zone(dist, db, floor=DEAD_DB):
    """Drop leading saturated/dead-zone samples (e.g. the 0.0; -49.9 launch point)."""
    k = 0
    while k < len(db) - 1 and db[k] < floor:
        k += 1
    return dist[k:], db[k:]


# --------------------------------------------------------------------------- #
# 2. RESAMPLING + CHANNELS  — fixed physical grid so positions are comparable.
# --------------------------------------------------------------------------- #
def resample(dist_km, db, grid_km):
    """Interpolate dB onto the common grid. np.interp clamps outside the support."""
    return np.interp(grid_km, dist_km, db)


def make_channels(db_on_grid):
    """Per-trace z-scored dB + per-trace z-scored derivative -> (2, L) array.

    Per-trace normalization removes absolute launch-level differences and makes the
    model + inference fully self-contained (no stored dataset statistics needed)."""
    def z(x):
        s = x.std()
        return (x - x.mean()) / (s + 1e-8)
    sig  = z(db_on_grid)
    grad = z(np.gradient(db_on_grid))
    return np.stack([sig, grad], axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. DATASET BUILD  — load everything, auto-detect grid, resample.
# --------------------------------------------------------------------------- #
def build_dataset(root=ROOT, prefixes=CLASS_PREFIXES, n_points=N_POINTS):
    raw, labels, paths = [], [], []
    all_files = glob.glob(os.path.join(root, "*.csv"))
    if not all_files:
        raise RuntimeError(f"No CSVs found in {root}")

    for fp in all_files:
        name = os.path.basename(fp).lower()
        label = next((lab for pref, lab in prefixes.items() if name.startswith(pref)), None)
        if label is None:
            print(f"[skip] no class prefix matched: {os.path.basename(fp)}")
            continue
        try:
            dist, db, _ = parse_otdr_csv(fp)
            dist, db = trim_dead_zone(dist, db)
            if len(dist) < 8:
                continue
            raw.append((dist, db))
            labels.append(label)
            paths.append(fp)
        except Exception as e:
            print(f"[skip] {os.path.basename(fp)}: {e}")

    if not raw:
        raise RuntimeError("No traces loaded — check ROOT path and CLASS_PREFIXES.")

    # Common physical grid: from 0 to the 99th-percentile window (robust to outliers).
    max_km = float(np.percentile([d[-1] for d, _ in raw], 99))
    grid_km = np.linspace(0.0, max_km, n_points)

    X = np.stack([make_channels(resample(d, v, grid_km)) for d, v in raw])  # (N,2,L)
    y = np.asarray(labels, dtype=np.int64)
    print(f"Loaded {len(y)} traces | grid 0–{max_km*1000:.1f} m @ {n_points} pts")
    for c, name in enumerate(CLASS_NAMES):
        print(f"   class {c} ({name}): {(y == c).sum()}")
    return X, y, grid_km, paths


class OTDRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i):  return self.X[i], self.y[i]


# --------------------------------------------------------------------------- #
# 4. MODEL  — 1D CNN ending in GAP + Linear so a CAM is recoverable.
# --------------------------------------------------------------------------- #
class OTDRNet(nn.Module):
    def __init__(self, n_classes=3, in_ch=2, width=32):
        super().__init__()
        def block(i, o, k=7):
            p = k // 2
            return nn.Sequential(
                nn.Conv1d(i, o, k, padding=p), nn.BatchNorm1d(o), nn.ReLU(inplace=True),
                nn.Conv1d(o, o, k, padding=p), nn.BatchNorm1d(o), nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )
        self.features = nn.Sequential(
            block(in_ch, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            block(width * 4, width * 4),   # last conv stack -> CAM source
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(width * 4, n_classes)

    def forward(self, x, return_features=False):
        feat = self.features(x)                 # (B, C, L')
        pooled = self.gap(feat).squeeze(-1)     # (B, C)
        out = self.fc(pooled)                   # (B, n_classes)
        if return_features:
            return out, feat
        return out


# --------------------------------------------------------------------------- #
# 5. TRAINING
# --------------------------------------------------------------------------- #
def train_model(X, y, epochs=60, batch=32, lr=1e-3, patience=12):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    dl_tr  = DataLoader(OTDRDataset(X_tr,  y_tr),  batch_size=batch, shuffle=True)
    dl_val = DataLoader(OTDRDataset(X_val, y_val), batch_size=batch)
    dl_te  = DataLoader(OTDRDataset(X_te,  y_te),  batch_size=batch)

    # inverse-frequency class weights for the imbalance (class 3 is smaller)
    counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    w = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=DEVICE)
    print("class weights:", w.cpu().numpy())

    model = OTDRNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=4)
    crit = nn.CrossEntropyLoss(weight=w)

    best_val, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        vloss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                vloss += crit(out, yb).item() * len(yb)
                correct += (out.argmax(1) == yb).sum().item()
                n += len(yb)
        vloss /= n
        sched.step(vloss)
        print(f"epoch {ep:02d} | val_loss {vloss:.4f} | val_acc {correct/n:.3f}")

        if vloss < best_val - 1e-4:
            best_val, best_state, bad = vloss, {k: v.cpu().clone()
                                                for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print("early stopping"); break

    model.load_state_dict(best_state)

    # held-out test report
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            preds += model(xb.to(DEVICE)).argmax(1).cpu().tolist()
            gts += yb.tolist()
    print("\n=== TEST ===")
    print(classification_report(gts, preds, target_names=CLASS_NAMES, digits=3))
    print(confusion_matrix(gts, preds))
    return model


# --------------------------------------------------------------------------- #
# 6. CAM LOCALIZATION  — where on the fibre is the event?
# --------------------------------------------------------------------------- #
def compute_cam(model, x_2L, target_class):
    """x_2L: (2, L) float array.  Returns CAM upsampled to length L, in [0,1]."""
    model.eval()
    xt = torch.from_numpy(x_2L[None]).to(DEVICE)
    with torch.no_grad():
        _, feat = model(xt, return_features=True)      # (1, C, L')
    w = model.fc.weight[target_class].to(feat.device)  # (C,)
    cam = torch.einsum("c,bcl->bl", w, feat)[0]        # (L',)
    cam = torch.relu(cam)
    cam = nn.functional.interpolate(cam[None, None], size=x_2L.shape[1],
                                    mode="linear", align_corners=False)[0, 0]
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.ptp() + 1e-8)
    return cam


def refine_event(db_on_grid, cam, grid_km, cls, cam_thresh=0.5):
    """Sharpen the CAM peak with the derivative and return (event_m, zone_m).

    bend  -> steepest non-reflective drop;
    break -> collapse to noise floor (steepest drop after the Fresnel peak)."""
    support = np.where(cam >= cam_thresh)[0]
    if support.size == 0:
        support = np.array([int(cam.argmax())])
    lo, hi = support.min(), support.max()
    grad = np.gradient(db_on_grid)
    seg = grad[lo:hi + 1]
    event_idx = lo + int(np.argmin(seg))           # most negative gradient = the drop
    to_m = lambda i: float(grid_km[i] * 1000.0)
    return to_m(event_idx), (to_m(lo), to_m(hi))


def predict_file(path, model, grid_km):
    """Full inference on one CSV. Returns a dict with class, confidence, event, zone."""
    dist, db, meta = parse_otdr_csv(path)
    dist, db = trim_dead_zone(dist, db)
    db_grid = resample(dist, db, grid_km)
    x = make_channels(db_grid)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x[None]).to(DEVICE))
        prob = torch.softmax(logits, 1)[0].cpu().numpy()
    cls = int(prob.argmax())

    result = {
        "file": os.path.basename(path),
        "class": cls,
        "class_name": CLASS_NAMES[cls],
        "confidence": float(prob[cls]),
        "prob": {CLASS_NAMES[i]: float(prob[i]) for i in range(3)},
        "db_grid": db_grid,
        "grid_m": grid_km * 1000.0,
        "event_m": None,
        "zone_m": None,
        "cam": None,
    }
    if cls != 0:  # localize only bend / break
        cam = compute_cam(model, x, cls)
        event_m, zone_m = refine_event(db_grid, cam, grid_km, cls)
        result.update(event_m=event_m, zone_m=zone_m, cam=cam)
    return result


def plot_localization(result, save=None):
    """Plot the trace and shade the detected event zone."""
    import matplotlib.pyplot as plt
    g, db = result["grid_m"], result["db_grid"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(g, db, lw=1.3, color="#1f3a5f")
    title = f"{result['file']}  →  {result['class_name'].upper()}  ({result['confidence']:.0%})"
    if result["event_m"] is not None:
        lo, hi = result["zone_m"]
        ax.axvspan(lo, hi, color="orange", alpha=0.25, label="event zone")
        ax.axvline(result["event_m"], color="red", lw=1.6, ls="--",
                   label=f"event @ {result['event_m']:.1f} m")
        ax.legend(loc="lower left")
        title += f"   |   event @ {result['event_m']:.1f} m"
    ax.set_xlabel("distance (m)"); ax.set_ylabel("dB"); ax.set_title(title)
    ax.grid(alpha=0.3); fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
    return fig


# --------------------------------------------------------------------------- #
# 7. SAVE / LOAD  — checkpoint carries the grid so inference stays consistent.
# --------------------------------------------------------------------------- #
def save_checkpoint(model, grid_km, path="otdr_model.pt"):
    torch.save({"state_dict": model.state_dict(),
                "grid_km": grid_km, "class_names": CLASS_NAMES}, path)
    print("saved", path)


def load_checkpoint(path="otdr_model.pt"):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = OTDRNet().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["grid_km"]


# --------------------------------------------------------------------------- #
# 8. EXAMPLE RUN  — uncomment the pieces you want.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    X, y, grid_km, paths = build_dataset()
    model = train_model(X, y)
    save_checkpoint(model, grid_km)

    # Inference + highlight on a single file:
    # model, grid_km = load_checkpoint()
    # r = predict_file(r"C:\path\to\some_trace.csv", model, grid_km)
    # print(r["class_name"], r["confidence"], "event_m:", r["event_m"])
    # plot_localization(r, save="localized.png")