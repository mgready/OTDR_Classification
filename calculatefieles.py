from pathlib import Path
from collections import Counter
import re

DATASET_DIR = Path(r"C:\Users\Magzhan\OTDR_Classification\OTDR_Classification\Dataset")

csv_files = list(DATASET_DIR.rglob("*.csv"))
class_counts = Counter()
unknown_files = []

for file in csv_files:
    match = re.match(r"(class\d+)", file.name.lower())

    if match:
        class_counts[match.group(1)] += 1
    else:
        unknown_files.append(file.name)

print(f"Total number of OTDR traces: {len(csv_files)}\n")

print("Number of traces per class:")
for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

if unknown_files:
    print("\nFiles without a recognised class label:")
    for name in unknown_files:
        print(name)