
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))
from otdr_common import parse_otdr_csv, trim_dead_zone

DEFAULT_FILE = r"C:\Users\Magzhan\OTDR_Classification\OTDR_Classification\Dataset\class3_2026_05_14_14_04_11.csv"


def moving_average(x, window=9):
    window = max(3, int(window) | 1)
    if len(x) < window:
        return x.astype(float)
    return np.convolve(x, np.ones(window, dtype=float) / window, mode="same")


def main():
    ap = argparse.ArgumentParser(description="Plot one clean OTDR reflectogram without event labels.")
    ap.add_argument("--file", default=DEFAULT_FILE, help="input OTDR CSV file")
    ap.add_argument("--out", default="single_reflectogram.png", help="output PNG filename")
    ap.add_argument("--window", type=int, default=9, help="moving-average smoothing window in samples")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    km, db, _ = parse_otdr_csv(args.file)
    km, db = trim_dead_zone(km, db)
    distance_m = km * 1000.0
    db_smooth = moving_average(db, args.window)

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.plot(distance_m, db, color="#AFCBE5", linewidth=0.75, alpha=0.85, label="Raw trace")
    ax.plot(distance_m, db_smooth, color="#1F4E70", linewidth=1.35, label="Smoothed trace")

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Signal level (dB)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower left", frameon=True)
    ax.margins(x=0.01)
    fig.tight_layout()

    fig.savefig(args.out, dpi=args.dpi)
    plt.show()
    print(f"Saved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()