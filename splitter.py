"""
splitter.py — stage 1.

Reads the flat dataset (class1_*.csv, class2_*.csv, class3_*.csv in one folder)
and copies files into a stratified train / val / test folder tree:

    split/
        train/class1/  train/class2/  train/class3/
        val/  ...
        test/ ...

Copies (does not move) the originals, so the source folder stays intact.
Run once:  python splitter.py
"""

import os, glob, shutil
from collections import Counter
from sklearn.model_selection import train_test_split

from otdr_common import label_from_filename, CLASS_DIRNAMES, SEED

# ---- config ---------------------------------------------------------------- #
SRC      = r"C:\Users\PCA\Desktop\OTDR_CLassification\Dataset"
OUT      = r"C:\Users\PCA\Desktop\OTDR_CLassification\split"
RATIOS   = (0.70, 0.15, 0.15)        # train / val / test
# ---------------------------------------------------------------------------- #


def main():
    files = glob.glob(os.path.join(SRC, "*.csv"))
    paths, labels = [], []
    for fp in files:
        lab = label_from_filename(fp)
        if lab is None:
            print(f"[skip] no class prefix: {os.path.basename(fp)}")
            continue
        paths.append(fp); labels.append(lab)
    print(f"{len(paths)} labelled files | class counts: {dict(Counter(labels))}")

    # stratified: first carve off train, then split the rest into val/test
    tr_p, rest_p, tr_y, rest_y = train_test_split(
        paths, labels, test_size=1 - RATIOS[0], stratify=labels, random_state=SEED)
    val_frac = RATIOS[1] / (RATIOS[1] + RATIOS[2])
    val_p, te_p, val_y, te_y = train_test_split(
        rest_p, rest_y, test_size=1 - val_frac, stratify=rest_y, random_state=SEED)

    splits = {"train": (tr_p, tr_y), "val": (val_p, val_y), "test": (te_p, te_y)}

    # fresh output tree
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for s in splits:
        for d in CLASS_DIRNAMES:
            os.makedirs(os.path.join(OUT, s, d), exist_ok=True)

    for s, (ps, ys) in splits.items():
        for fp, lab in zip(ps, ys):
            dst = os.path.join(OUT, s, CLASS_DIRNAMES[lab], os.path.basename(fp))
            shutil.copy2(fp, dst)
        print(f"{s:5s}: {len(ps):4d} files | {dict(Counter(ys))}")

    print(f"\ndone -> {OUT}")


if __name__ == "__main__":
    main()
