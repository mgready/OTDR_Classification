# OTDR Trend Analysis & Classification

A physics-driven pipeline that decomposes OTDR reflectograms into **trend lines**
(piecewise-linear segments) and **events** (bend / connector / break / end), then
classifies each trace and lets you view, annotate, and label traces in a desktop GUI.

The approach is **length-independent**: it works on the raw `(distance, dB)` trace using
distance-based windows, so the same code handles a 150 m lab fibre or a multi-km field span.

---

## 1. Requirements

| Component | Version |
|-----------|---------|
| Python    | 3.9 – 3.12 |
| OS        | Windows / macOS / Linux (GUI tested on Windows) |

### Python libraries

```bash
pip install numpy scipy scikit-learn pandas matplotlib joblib PyQt5
```

| Library | Used by |
|---------|---------|
| numpy, scipy | all trend/analysis code |
| scikit-learn, pandas, joblib | clustering & training (`otdr_cluster.py`, `train_trend_clf.py`) |
| matplotlib | all plots and the GUI canvas |
| PyQt5 | the GUI only (`otdr_studio.py`) |

> `torch` is **only** needed for the legacy CNN scripts (see §7). The trend pipeline does not use it.

### Using conda (recommended)

```bash
conda create -n otdr python=3.11
conda activate otdr
pip install numpy scipy scikit-learn pandas matplotlib joblib PyQt5
```

---

## 2. Dataset format

CSV files, **semicolon-separated** with **comma decimals** (European format), a small
metadata header, then a `km;dB` data block:

```
pulse width[0] (ns);6
n;1,468100
number of samples;627
km;dB
0,000000;-49,900002
0,000319;-8,139000
...
```

**Folder layout** — all CSVs in one folder, labelled by filename prefix:

```
Dataset/
    class1_*.csv      # normal
    class2_*.csv      # bend
    class3_*.csv      # break
    class4_*.csv      # (optional 4th category)
```

The parser is tolerant of header variations — any row whose first two fields are numbers is
treated as data.

---

## 3. File map

| File | Role | Run directly? |
|------|------|---------------|
| `otdr_common.py`  | Shared library: CSV parsing, dead-zone trim, filename→label. | No (imported) |
| `otdr_trends.py`  | Piecewise trend decomposition, region detection, event extraction. | Optional (1 file) |
| `otdr_channel.py` | Regression-channel detector + rule-based class verdict. | Optional (1 file) |
| `otdr_cluster.py` | Trend features → clustering (discover bend/break sub-types). | **Yes** |
| `train_trend_clf.py` | Supervised training + testing; saves the model. | **Yes** |
| `detector_trained.py` | Wraps the trained model as a GUI detector plugin. | No (loaded in GUI) |
| `otdr_studio.py`  | PyQt5 GUI: viewer, trend overlay, ignore zones, annotation, zoom. | **Yes** |
| `batch_channel.py` | Batch rule-based classifier + feature CSV (sanity check). | Optional |

All trend scripts import the shared helpers from `otdr_common.py`, so **keep every file in
the same folder.**

---

## 4. Where to change paths

Every script has a small config block at the top. **Replace the example folder
`C:\Users\PCA\Desktop\OTDR_CLassification` with your own path.**

| File | Constant(s) to edit | Meaning |
|------|--------------------|---------|
| `otdr_cluster.py` | `ROOT`, `OUT_CSV`, `PROJ_PNG`, `EXPECTED_M` | dataset folder, output CSV/PNG, OTDR length (m) |
| `train_trend_clf.py` | `ROOT`, `MODEL_PATH`, `TREE_PNG`, `EXPECTED_M`, `REGION_METHOD` | dataset, saved model, tree image, length, noise method |
| `detector_trained.py` | `MODEL_PATH` | must match `train_trend_clf.py`'s `MODEL_PATH` |
| `batch_channel.py` | `ROOT`, `FEATURE_CSV`, `EXPECTED_LEN_M` | dataset, output CSV, length |
| `otdr_trends.py` / `otdr_channel.py` | default path in the `__main__` block | only the demo file when run standalone |
| `otdr_studio.py` | none — uses the **Open folder…** dialog | — |

### Two settings that must be consistent

- **`EXPECTED_M` (OTDR / fibre length in metres).** This is the reference for `terminal_frac`
  (how far the fibre survived vs its expected length). A collapse near this length = normal end;
  an early collapse = break. **Use the same value when training and when viewing in the GUI.**
- **`REGION_METHOD`** (`"variance"`, `"level"`, or `"gradient"`) — how the noise floor / valid
  region is detected. **Train and infer with the same method.**

---

## 5. How to run (workflow)

```bash
# 1. Explore the natural pattern types (clustering)
python otdr_cluster.py
#    -> trend_clusters.png, trend_cluster_features.csv

# 2. Train + test the classifier
python train_trend_clf.py
#    -> prints CV accuracy, TEST confusion matrix, decision-tree rules
#    -> trend_tree.png, trend_clf.joblib  (the model)

# 3. Launch the GUI
python otdr_studio.py
```

Inside the GUI:

1. **Open folder…** → choose your `Dataset` folder; click a file or use Prev/Next.
2. **OTDR length (m)** → type your cable length (the reference for break detection).
3. **Trend detector** → pick `Piecewise Linear` (trends) or `Load custom model…` and select
   `detector_trained.py` to run the **trained classifier** live.
4. **region method** → `variance` / `level` / `gradient` (changes the shaded valid region).
5. **Smoothing** → None / Moving average / Savitzky-Golay / Median / Gaussian.
6. **Ignore zones** → drag on the plot to exclude a span from the trend fit.
7. **Annotation (ML mask)** → click to drop labelled points; **Save mask** writes a per-file
   JSON for ML training.
8. **Zoom / pan** → the Matplotlib toolbar at the top; **Fit valid region** to snap the view.

---

## 6. Retraining

The model is saved at `MODEL_PATH` (`trend_clf.joblib`). **Retrain whenever you change**:
- the dataset, `EXPECTED_M`, or `REGION_METHOD`;
- the feature/merge logic in `otdr_trends.py` (e.g. event merging, terminus anchoring).

```bash
python train_trend_clf.py     # rebuilds features and overwrites trend_clf.joblib
```

`detector_trained.py` loads whatever is at `MODEL_PATH`, so the GUI uses the latest model
automatically after retraining.

---

## 7. Legacy (optional) — CNN classifier

The earlier 1-D CNN approach (`splitter.py`, `classifier.py`, `checker.py`,
`otdr_classifier.py`) is superseded by the trend pipeline and is **not required**. It needs
PyTorch:

```bash
pip install torch
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Weights only load failed` on `torch.load` | PyTorch ≥ 2.6 default; load with `weights_only=False` (legacy CNN only). |
| `'ptp' was removed from the ndarray class` | NumPy ≥ 2.0; use `np.ptp(arr)` not `arr.ptp()`. |
| `No numeric data rows found` | A CSV uses a different delimiter/format — open it and check it's `;`-separated with `,` decimals. |
| `from __future__ import annotations` SyntaxError | That line must sit directly under the docstring, or just delete it. |
| KMeans OpenMP warning on Windows | Harmless; already silenced via `OMP_NUM_THREADS=3`. |
| GUI won't start | `pip install PyQt5`; ensure all `.py` files are in one folder. |
| Same-looking traces classify differently | The class may be missing from the training set, or the OTDR length differs between training and the GUI — keep `EXPECTED_M` consistent and retrain. |

---

## 9. Quick reference — pipeline at a glance

```
CSV ─▶ parse (otdr_common)
    ─▶ find valid region + events (otdr_trends)
    ─▶ trend features (otdr_cluster.trace_features)
    ─▶ classifier (train_trend_clf → trend_clf.joblib)
    ─▶ live in GUI (otdr_studio + detector_trained)
```
