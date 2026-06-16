"""
classifier.py — stage 2.  TRAIN + VALIDATION.

Trains on split/train, early-stops on split/val, saves otdr_model.pt
(weights + resampling grid). Does NOT touch the test set.

Run:  python classifier.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from otdr_common import (
    load_raw_folder, compute_grid, traces_to_X, OTDRDataset, OTDRNet,
    save_checkpoint, CLASS_NAMES, DEVICE)

# ---- config ---------------------------------------------------------------- #
SPLIT   = r"C:\Users\PCA\Desktop\OTDR_CLassification\split"
CKPT    = r"C:\Users\PCA\Desktop\OTDR_CLassification\otdr_model.pt"
EPOCHS, BATCH, LR, PATIENCE = 60, 32, 1e-3, 12
# ---------------------------------------------------------------------------- #


def main():
    # load raw traces; grid is fixed from TRAIN only, then reused for val
    raw_tr  = load_raw_folder(os.path.join(SPLIT, "train"))
    raw_val = load_raw_folder(os.path.join(SPLIT, "val"))
    if not raw_tr or not raw_val:
        raise RuntimeError("train/ or val/ empty — run splitter.py first.")

    grid = compute_grid(raw_tr)
    X_tr, y_tr, _ = traces_to_X(raw_tr, grid)
    X_val, y_val, _ = traces_to_X(raw_val, grid)
    print(f"train {len(y_tr)} | val {len(y_val)} | grid 0-{grid[-1]*1000:.1f} m @ {len(grid)} pts")
    for c, n in enumerate(CLASS_NAMES):
        print(f"   {n}: train {(y_tr==c).sum()} / val {(y_val==c).sum()}")

    dl_tr  = DataLoader(OTDRDataset(X_tr,  y_tr),  batch_size=BATCH, shuffle=True)
    dl_val = DataLoader(OTDRDataset(X_val, y_val), batch_size=BATCH)

    # inverse-frequency class weights (class3/break is the minority)
    counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    w = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=DEVICE)
    print("class weights:", np.round(w.cpu().numpy(), 3))

    model = OTDRNet().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=4)
    crit  = nn.CrossEntropyLoss(weight=w)

    best, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()

        model.eval()
        vloss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                vloss += crit(out, yb).item() * len(yb)
                correct += (out.argmax(1) == yb).sum().item(); n += len(yb)
        vloss /= n
        sched.step(vloss)
        print(f"epoch {ep:02d} | val_loss {vloss:.4f} | val_acc {correct/n:.3f}")

        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stopping"); break

    model.load_state_dict(best_state)

    # validation report (final)
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl_val:
            preds += model(xb.to(DEVICE)).argmax(1).cpu().tolist(); gts += yb.tolist()
    print("\n=== VALIDATION (best model) ===")
    print(classification_report(gts, preds, target_names=CLASS_NAMES, digits=3))
    print(confusion_matrix(gts, preds))

    save_checkpoint(model, grid, CKPT)


if __name__ == "__main__":
    main()
