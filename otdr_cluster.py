"""
otdr_cluster.py — discover OTDR pattern types by clustering trend features.

Each trace's piecewise decomposition (segments + typed events) becomes a fixed feature
vector; we then standardize, project to 2D (PCA), and cluster (KMeans, k chosen by
silhouette). The cluster x true-class crosstab shows how many distinct BEND types and
BREAK types actually exist in the data.

Key distance-aware feature: terminal_frac = (where the fibre effectively ends) / (expected
length). A normal end and the commonest break have the SAME shape; only their position
relative to the expected length differs. This feature makes "same end-pattern at 90 m on a
150 m fibre = break" separable, exactly as you described.

Run:  python otdr_cluster.py
"""

import os, glob
os.environ.setdefault("OMP_NUM_THREADS", "3")    # silences sklearn KMeans Windows warning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from otdr_common import label_from_filename
from otdr_trends import analyze_file

# ---- config ---------------------------------------------------------------- #
ROOT       = r"C:\Users\PCA\Desktop\OTDR_CLassification\Dataset"
OUT_CSV    = r"C:\Users\PCA\Desktop\OTDR_CLassification\trend_cluster_features.csv"
PROJ_PNG   = r"C:\Users\PCA\Desktop\OTDR_CLassification\trend_clusters.png"
EXPECTED_M = 150.0
NAMES      = {1: "normal", 2: "bend", 3: "break"}
FEAT_COLS  = ["terminal_frac", "backscatter_slope", "max_loss_dB", "total_loss_dB",
              "n_bend", "terminal_reflect_dB", "first_loss_frac"]
# ---------------------------------------------------------------------------- #


def trace_features(out, expected_m):
    """Robust, physically-bounded features from one piecewise decomposition.

    Avoids raw per-segment slope stats (tiny segments across reflections give absurd
    slopes that dominate clustering). Uses only the dominant backscatter slope, clipped,
    plus distance-aware and loss/reflection features."""
    km, segs, evs, eof = out["km"], out["segments"], out["events"], out["eof"]
    terminal_m = float(km[eof] * 1000.0)

    # main backscatter slope = slope of the LONGEST segment, clipped to a sane range
    if segs:
        longest = max(segs, key=lambda s: s["i1"] - s["i0"])
        main_slope = float(np.clip(longest["slope_dB_per_km"], -20.0, 5.0))
    else:
        main_slope = 0.0

    bends = [e for e in evs if e["type"] == "bend"]
    term_refl = max([e.get("reflectance_dB", 0.0) for e in evs
                     if abs(e["m"] - terminal_m) < 0.05 * expected_m], default=0.0)
    first_loss = min([e["m"] for e in bends], default=terminal_m)

    return {
        "terminal_frac":       min(terminal_m / expected_m, 1.5),    # distance-aware (key)
        "terminal_m":          round(terminal_m, 1),
        "backscatter_slope":   main_slope,
        "max_loss_dB":         float(min(max([e["loss_dB"] for e in bends], default=0.0), 25.0)),
        "total_loss_dB":       float(min(sum(max(0.0, e["loss_dB"]) for e in bends), 40.0)),
        "n_bend":              int(min(len(bends), 6)),
        "terminal_reflect_dB": float(min(term_refl, 15.0)),
        "first_loss_frac":     min(first_loss / expected_m, 1.2),
    }


def main():
    rows = []
    for fp in glob.glob(os.path.join(ROOT, "*.csv")):
        lab = label_from_filename(fp)
        if lab is None:
            continue
        try:
            out = analyze_file(fp, expected_length_m=EXPECTED_M)
            f = trace_features(out, EXPECTED_M)
        except Exception as e:
            print(f"[skip] {os.path.basename(fp)}: {e}")
            continue
        f["file"] = os.path.basename(fp)
        f["true_class"] = lab + 1
        rows.append(f)

    df = pd.DataFrame(rows).fillna(0.0)
    print(f"{len(df)} traces -> {len(FEAT_COLS)} features each")
    X = StandardScaler().fit_transform(df[FEAT_COLS].values)

    # pick k by silhouette
    best_k, best_s = 3, -1.0
    for k in range(3, 9):
        lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        s = silhouette_score(X, lab)
        print(f"k={k}  silhouette={s:.3f}")
        if s > best_s:
            best_k, best_s = k, s
    df["cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X)
    print(f"\nchosen k = {best_k}")

    print("\ncluster x true-class (how many bend/break sub-types exist):")
    print(pd.crosstab(df["cluster"], df["true_class"].map(NAMES)))

    print("\nper-cluster feature means (read these to NAME each type):")
    print(df.groupby("cluster")[FEAT_COLS].mean().round(2).T)

    # 2D projection: true class vs discovered cluster
    p = PCA(2).fit_transform(X)
    df["pc1"], df["pc2"] = p[:, 0], p[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for c in sorted(df["true_class"].unique()):
        m = df["true_class"] == c
        ax[0].scatter(df.pc1[m], df.pc2[m], s=18, alpha=0.7, label=NAMES[c])
    ax[0].set_title("colored by TRUE class")
    for c in sorted(df["cluster"].unique()):
        m = df["cluster"] == c
        ax[1].scatter(df.pc1[m], df.pc2[m], s=18, alpha=0.7, label=f"cluster {c}")
    ax[1].set_title(f"colored by KMeans cluster (k={best_k})")
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PROJ_PNG, dpi=130)

    df.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {PROJ_PNG}\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()