from pathlib import Path
from collections import defaultdict
import json
import random
import shutil

SOURCE_DIR = Path(r"C:\Users\Magzhan\OTDR_Classification\OTDR_Classification\Dataset")
OUTPUT_DIR = SOURCE_DIR.parent / "Dataset_annotated_split"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
SEED = 42

random.seed(SEED)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {SOURCE_DIR}")
if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
    raise ValueError("Split ratios must sum to 1.0")
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output folder already exists: {OUTPUT_DIR}\n"
        "Delete it or change OUTPUT_DIR before running the script."
    )

# Use only CSV traces that have a valid paired .mask.json file.
# The mask remains available for feature creation and validation; it is not copied into test.
annotated = []
skipped = []

for csv_path in sorted(SOURCE_DIR.rglob("*.csv")):
    mask_path = csv_path.with_suffix(".mask.json")
    if not mask_path.exists():
        skipped.append((csv_path.name, "mask file is missing"))
        continue

    try:
        with mask_path.open(encoding="utf-8") as f:
            mask = json.load(f)
        if mask.get("file") not in (None, csv_path.name):
            skipped.append((csv_path.name, "mask 'file' value does not match CSV name"))
            continue
        if "annotations" not in mask:
            skipped.append((csv_path.name, "mask has no annotations field"))
            continue
    except (OSError, json.JSONDecodeError) as error:
        skipped.append((csv_path.name, f"invalid mask: {error}"))
        continue

    group = csv_path.name.split("_")[0].lower()
    annotated.append((group, csv_path, mask_path))

if not annotated:
    raise RuntimeError("No valid CSV + .mask.json pairs were found.")

by_group = defaultdict(list)
for group, csv_path, mask_path in annotated:
    by_group[group].append((csv_path, mask_path))

manifest = []
for group in sorted(by_group):
    pairs = sorted(by_group[group], key=lambda x: x[0].name)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)
    n_test = n - n_train - n_val

    partitions = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for split_name, split_pairs in partitions.items():
        destination = OUTPUT_DIR / split_name / group
        destination.mkdir(parents=True, exist_ok=True)

        for csv_path, mask_path in split_pairs:
            # CSV files are copied to every split.
            shutil.copy2(csv_path, destination / csv_path.name)

            # Ground truth is usable for train/validation only.
            # Test masks are deliberately excluded to enforce blind prediction.
            if split_name in {"train", "val"}:
                shutil.copy2(mask_path, destination / mask_path.name)

            manifest.append({
                "split": split_name,
                "acquisition_group": group,
                "csv_file": csv_path.name,
                "mask_available_to_model": split_name in {"train", "val"},
                "mask_source_file": mask_path.name,
            })

    print(
        f"{group}: annotated={n}, train={len(partitions['train'])}, "
        f"val={len(partitions['val'])}, test={len(partitions['test'])}"
    )

manifest_path = OUTPUT_DIR / "split_manifest.csv"
with manifest_path.open("w", encoding="utf-8") as f:
    f.write("split,acquisition_group,csv_file,mask_available_to_model,mask_source_file\n")
    for row in manifest:
        f.write(
            f"{row['split']},{row['acquisition_group']},{row['csv_file']},"
            f"{row['mask_available_to_model']},{row['mask_source_file']}\n"
        )

print(f"\nValid annotated traces: {len(annotated)}")
print(f"Traces skipped because no valid mask was available: {len(skipped)}")
print(f"Split manifest: {manifest_path}")
print(f"Output directory: {OUTPUT_DIR}")
print("\nImportant: test contains CSV files only. Keep the original test masks outside this folder and use them only after prediction for final evaluation.")
