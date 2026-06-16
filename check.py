"""
check_model.py — load otdr_model.pt and inspect predictions + localization.

Run:  python check_model.py
It will:
  1. sweep every CSV in the dataset folder, predict, and print accuracy + confusion
     matrix (NOTE: this mixes train/val/test, so it is optimistic — use it to eyeball
     behaviour, not to report a score; the held-out test report from training is the
     real metric);
  2. save localization plots for a few bend/break examples so you can verify the
     highlighted zone sits on the actual event.

Requires otdr_classifier.py in the same folder.
"""

import os
import glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from otdr_classifier import (
    ROOT, CLASS_PREFIXES, CLASS_NAMES,
    load_checkpoint, predict_file, plot_localization,
)

CKPT     = r"C:\Users\PCA\Desktop\OTDR_CLassification\otdr_model.pt"
PLOT_DIR = r"C:\Users\PCA\Desktop\OTDR_CLassification\checks"
N_PLOTS_PER_CLASS = 4          # how many example plots to save per class


def label_from_name(path):
    name = os.path.basename(path).lower()
    return next((lab for p, lab in CLASS_PREFIXES.items() if name.startswith(p)), None)


def main():
    model, grid_km = load_checkpoint(CKPT)
    files = glob.glob(os.path.join(ROOT, "*.csv"))
    print(f"found {len(files)} CSVs\n")

    gts, preds, rows = [], [], []
    for fp in files:
        lab = label_from_name(fp)
        if lab is None:
            continue
        try:
            r = predict_file(fp, model, grid_km)
        except Exception as e:
            print(f"[skip] {os.path.basename(fp)}: {e}")
            continue
        gts.append(lab)
        preds.append(r["class"])
        rows.append((fp, lab, r))

    # ---- overall numbers (optimistic: includes training files) ----------------
    print("=== whole-dataset sweep (optimistic, includes train) ===")
    print(classification_report(gts, preds, target_names=CLASS_NAMES, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(gts, preds), "\n")

    # ---- show a few misclassified files, if any -------------------------------
    wrong = [(os.path.basename(fp), CLASS_NAMES[t], CLASS_NAMES[r['class']], r['confidence'])
             for fp, t, r in rows if t != r["class"]]
    if wrong:
        print(f"{len(wrong)} misclassified files (true -> pred, conf):")
        for nm, t, p, c in wrong[:25]:
            print(f"   {nm:45s} {t:7s} -> {p:7s}  ({c:.0%})")
    else:
        print("no misclassifications on the sweep.")

    # ---- save localization plots for a handful per class ----------------------
    os.makedirs(PLOT_DIR, exist_ok=True)
    saved = {0: 0, 1: 0, 2: 0}
    for fp, lab, r in rows:
        if saved[lab] >= N_PLOTS_PER_CLASS:
            continue
        out = os.path.join(PLOT_DIR, f"{CLASS_NAMES[lab]}_{saved[lab]}_{os.path.basename(fp)}.png")
        plot_localization(r, save=out)
        saved[lab] += 1
        ev = f"event @ {r['event_m']:.1f} m" if r["event_m"] is not None else "—"
        print(f"saved {os.path.basename(out)}  | pred={r['class_name']} ({r['confidence']:.0%})  {ev}")
    print(f"\nplots in: {PLOT_DIR}")


if __name__ == "__main__":
    main()