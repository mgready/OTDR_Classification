"""
checker.py — stage 3.  TEST.

Loads otdr_model.pt and evaluates on split/test (held out — never seen in
training/validation). Prints the real classification report + confusion matrix,
lists misclassifications, and saves localization plots so you can verify the
highlighted bend/break zone sits on the actual event.

Run:  python checker.py
"""

import os, glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from otdr_common import (
    load_checkpoint, predict_file, plot_localization,
    CLASS_DIRNAMES, CLASS_NAMES)

# ---- config ---------------------------------------------------------------- #
SPLIT    = r"C:\Users\PCA\Desktop\OTDR_CLassification\split"
CKPT     = r"C:\Users\PCA\Desktop\OTDR_CLassification\otdr_model.pt"
PLOT_DIR = r"C:\Users\PCA\Desktop\OTDR_CLassification\checks"
N_PLOTS_PER_CLASS = 5
# ---------------------------------------------------------------------------- #


def main():
    model, grid = load_checkpoint(CKPT)
    test_dir = os.path.join(SPLIT, "test")

    gts, preds, rows = [], [], []
    for label, dirn in enumerate(CLASS_DIRNAMES):
        for fp in glob.glob(os.path.join(test_dir, dirn, "*.csv")):
            try:
                r = predict_file(fp, model, grid)
            except Exception as e:
                print(f"[skip] {os.path.basename(fp)}: {e}"); continue
            gts.append(label); preds.append(r["class"]); rows.append((fp, label, r))

    if not rows:
        raise RuntimeError("test/ empty — run splitter.py then classifier.py first.")

    print(f"=== TEST ({len(rows)} files, held out) ===")
    print(classification_report(gts, preds, labels=[0, 1, 2],
                            target_names=CLASS_NAMES, digits=3, zero_division=0))
    print(confusion_matrix(gts, preds, labels=[0, 1, 2]), "\n")
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(gts, preds), "\n")

    wrong = [(os.path.basename(fp), CLASS_NAMES[t], CLASS_NAMES[r['class']], r['confidence'])
             for fp, t, r in rows if t != r["class"]]
    if wrong:
        print(f"{len(wrong)} misclassified:")
        for nm, t, p, c in wrong:
            print(f"   {nm:45s} {t:7s} -> {p:7s} ({c:.0%})")
    else:
        print("no misclassifications on the test set.")

    # localization plots
    os.makedirs(PLOT_DIR, exist_ok=True)
    saved = {0: 0, 1: 0, 2: 0}
    for fp, lab, r in rows:
        if saved[lab] >= N_PLOTS_PER_CLASS:
            continue
        out = os.path.join(PLOT_DIR, f"{CLASS_NAMES[lab]}_{saved[lab]}_{os.path.basename(fp)}.png")
        plot_localization(r, save=out)
        saved[lab] += 1
        ev = f"event @ {r['event_m']:.1f} m" if r["event_m"] is not None else "—"
        print(f"saved {os.path.basename(out)} | pred={r['class_name']} ({r['confidence']:.0%}) {ev}")
    print(f"\nplots in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
