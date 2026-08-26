from pathlib import Path
from collections import Counter
import json
import random
import shutil

SOURCE_DIR = Path(r"C:\Users\Magzhan\OTDR_Classification\OTDR_Classification\Dataset")
OUTPUT_DIR = SOURCE_DIR.parent / "Dataset_event_stratified_split"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
SEED = 42
EVENTS = ("bend", "connector", "break")

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")
if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {SOURCE_DIR}")
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output folder already exists: {OUTPUT_DIR}\n"
        "Delete it or change OUTPUT_DIR before running the script."
    )

rng = random.Random(SEED)
records = []
skipped = []

# A record represents one whole OTDR trace. Only files with a valid GT mask are eligible.
for csv_path in sorted(SOURCE_DIR.rglob("*.csv")):
    mask_path = csv_path.with_suffix(".mask.json")
    if not mask_path.exists():
        skipped.append((csv_path.name, "missing mask"))
        continue

    try:
        with mask_path.open(encoding="utf-8") as f:
            mask = json.load(f)
    except (OSError, json.JSONDecodeError) as error:
        skipped.append((csv_path.name, f"invalid mask: {error}"))
        continue

    if mask.get("file") not in (None, csv_path.name):
        skipped.append((csv_path.name, "mask file name mismatch"))
        continue

    labels = {
        str(annotation.get("type", "")).strip().lower()
        for annotation in mask.get("annotations", [])
        if str(annotation.get("type", "")).strip().lower() in EVENTS
    }

    records.append({
        "csv": csv_path,
        "mask": mask_path,
        "labels": labels,
        "acquisition_group": csv_path.name.split("_")[0].lower(),
    })

if not records:
    raise RuntimeError("No valid CSV + .mask.json pairs found.")

# Iterative greedy multi-label allocation.
# Rare event labels are allocated first, while each split is kept close to 70/10/20.
split_names = ("train", "val", "test")
ratios = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}
assignments = {name: [] for name in split_names}
assigned_ids = set()

label_totals = Counter(label for r in records for label in r["labels"])
target_label_counts = {
    split: {event: label_totals[event] * ratios[split] for event in EVENTS}
    for split in split_names
}
target_file_counts = {split: len(records) * ratios[split] for split in split_names}
current_label_counts = {split: Counter() for split in split_names}
current_file_counts = Counter()

# Files with rare labels and more labels are placed first.
ordered = sorted(
    enumerate(records),
    key=lambda item: (
        min((label_totals[label] for label in item[1]["labels"]), default=10**9),
        -len(item[1]["labels"]),
        item[1]["csv"].name,
    ),
)

for record_id, record in ordered:
    scores = {}
    for split in split_names:
        # Reward filling missing event-label targets.
        label_need = sum(
            max(0.0, target_label_counts[split][label] - current_label_counts[split][label])
            / max(1.0, target_label_counts[split][label])
            for label in record["labels"]
        )

        # Keep total numbers of traces close to requested proportions.
        file_need = max(0.0, target_file_counts[split] - current_file_counts[split]) / max(
            1.0, target_file_counts[split]
        )

        # Small random tie-breaker with a fixed seed for reproducibility.
        scores[split] = 3.0 * label_need + file_need + rng.random() * 1e-6

    chosen = max(scores, key=scores.get)
    assignments[chosen].append(record)
    assigned_ids.add(record_id)
    current_file_counts[chosen] += 1
    current_label_counts[chosen].update(record["labels"])

assert sum(len(items) for items in assignments.values()) == len(records)

# Copy train/validation masks; intentionally do not copy test masks.
manifest_rows = []
for split, items in assignments.items():
    for record in sorted(items, key=lambda r: r["csv"].name):
        destination = OUTPUT_DIR / split
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["csv"], destination / record["csv"].name)

        if split in {"train", "val"}:
            shutil.copy2(record["mask"], destination / record["mask"].name)

        manifest_rows.append({
            "split": split,
            "csv_file": record["csv"].name,
            "source_acquisition_group": record["acquisition_group"],
            "has_bend": int("bend" in record["labels"]),
            "has_connector": int("connector" in record["labels"]),
            "has_break": int("break" in record["labels"]),
            "mask_copied": split in {"train", "val"},
            "original_mask": record["mask"].name,
        })

# Write an audit manifest and a concise distribution report.
manifest_path = OUTPUT_DIR / "split_manifest.csv"
columns = [
    "split", "csv_file", "source_acquisition_group", "has_bend",
    "has_connector", "has_break", "mask_copied", "original_mask",
]
with manifest_path.open("w", encoding="utf-8", newline="") as f:
    f.write(",".join(columns) + "\n")
    for row in sorted(manifest_rows, key=lambda x: (x["split"], x["csv_file"])):
        f.write(",".join(str(row[column]) for column in columns) + "\n")

print(f"Eligible annotated traces: {len(records)}")
print(f"Skipped CSV files without a valid mask: {len(skipped)}\n")
print("Event-presence distribution by split (number of traces containing each event):")
for split in split_names:
    items = assignments[split]
    counts = Counter(label for r in items for label in r["labels"])
    normal = sum(not r["labels"] for r in items)
    print(
        f"{split:5s}: files={len(items):3d} | bend={counts['bend']:3d} | "
        f"connector={counts['connector']:3d} | break={counts['break']:3d} | normal={normal:3d}"
    )

print(f"\nCreated split at: {OUTPUT_DIR}")
print(f"Manifest: {manifest_path}")
print("Train and validation contain CSV + .mask.json files.")
print("Test contains CSV files only. Keep original test masks in the source Dataset folder and use them only after predictions are finalized.")
