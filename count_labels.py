"""
count_labels.py — простой подсчёт классов во всех .mask.json датасета.

Проходит по всем *.mask.json в папке, суммирует количество вхождений
каждого type в поле annotations (если один тип встретился в файле дважды —
считается дважды). Никакой ML-логики, никакого noise, никаких breakpoint-
детекторов — только то, что реально лежит в JSON.

Запуск:
    python count_labels.py --data "C:\\Users\\PCA\\OneDrive - Astana IT University\\Desktop\\OTDR_CLassification\\Dataset"
"""

import os
import glob
import json
import argparse
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="папка с .mask.json файлами")
    args = ap.parse_args()

    mask_files = sorted(glob.glob(os.path.join(args.data, "*.mask.json")))
    print(f"Найдено .mask.json файлов: {len(mask_files)}")

    counter = Counter()
    files_with_no_annotations = 0
    broken_files = []

    for mf in mask_files:
        try:
            data = json.load(open(mf, encoding="utf-8"))
            annos = data.get("annotations", [])
            if not annos:
                files_with_no_annotations += 1
            for a in annos:
                t = a.get("type", "UNKNOWN")
                counter[t] += 1
        except Exception as e:
            broken_files.append((os.path.basename(mf), str(e)))

    print("\n=== Подсчёт классов (по всем annotations во всех файлах) ===")
    for label, count in counter.most_common():
        print(f"  {label:15s} {count}")

    print(f"\nВсего событий: {sum(counter.values())}")
    print(f"Файлов без единой аннотации: {files_with_no_annotations}")

    if broken_files:
        print(f"\nНе удалось прочитать {len(broken_files)} файлов:")
        for name, err in broken_files:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()