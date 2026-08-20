"""
balance_candidate_data.py — балансирует candidate-level train/validation dataset.

Вход (создан build_candidate_train_val.py):
  candidate_event_model_data/train_candidates.csv
  candidate_event_model_data/val_candidates.csv

Проблема исходных CSV:
  train: 25,952 background vs ~656 real candidate-positive
  val:    8,736 background vs ~225 real candidate-positive

На таком датасете большинство моделей почти всегда выберут background.

Этот скрипт:
  - СОХРАНЯЕТ ВСЕ positive candidates: bend / connector / break.
  - Семплирует только background.
  - Берёт `background_per_positive` negatives на один positive (default=4).
  - 70% hard negatives: точки с сильными event-like признаками; именно они
    важны для снижения FP в реальном инференсе.
  - 30% random negatives: обычные фоновые точки для разнообразия.
  - Train и val обрабатываются отдельно; файлы НЕ смешиваются, split не меняется.

Важно:
  Этот скрипт НЕ решает candidate recall. Пропущенные GT, которых нет среди
  candidates, останутся структурным ограничением recall. Он решает дисбаланс
  и делает обучение классификатора осмысленным.

Запуск:
  python balance_candidate_data.py

Опции:
  python balance_candidate_data.py --background-per-positive 5 --hard-fraction 0.75
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IN = SCRIPT_DIR / "candidate_event_model_data"
DEFAULT_OUT = SCRIPT_DIR / "candidate_event_model_data_balanced"
EVENT_CLASSES = {"bend", "connector", "break"}


def robust_z(series):
    """Robust normalized score; keeps hard-negative ranking stable despite outliers."""
    x = series.astype(float).abs().fillna(0.0)
    med = x.median()
    mad = (x - med).abs().median()
    return (x - med) / (1.4826 * mad + 1e-8)


def hard_negative_score(bg_df):
    """Высокий score = background, который выглядит как настоящее OTDR-событие.

    Такие points особенно важны: если их не показать модели на train, они
    превращаются в ложные bend/break/connector на test.
    """
    score = (
        1.5 * robust_z(bg_df["loss_dB"]) +
        1.2 * robust_z(bg_df["peak_above_bg_dB"]) +
        1.0 * robust_z(bg_df["local_range_db"]) +
        0.8 * robust_z(bg_df["local_std_db"]) +
        0.7 * robust_z(bg_df["max_pre_derivative"]) +
        0.5 * robust_z(bg_df["post_std_db"])
    )
    return score


def balance_one(df, background_per_positive, hard_fraction, seed, split_name):
    df = df.copy()
    positives = df[df["label"].isin(EVENT_CLASSES)].copy()
    background = df[df["label"] == "background"].copy()

    if positives.empty:
        raise ValueError(f"{split_name}: no positive candidate rows")
    if background.empty:
        raise ValueError(f"{split_name}: no background rows")

    n_pos = len(positives)
    target_bg = min(len(background), int(round(n_pos * background_per_positive)))
    n_hard = min(len(background), int(round(target_bg * hard_fraction)))
    n_random = max(0, target_bg - n_hard)

    background["hard_negative_score"] = hard_negative_score(background)
    hard_pool = background.sort_values("hard_negative_score", ascending=False)
    hard = hard_pool.head(n_hard).copy()

    # Random pool excludes already selected hard rows.
    remaining = background.drop(index=hard.index)
    random = remaining.sample(n=min(n_random, len(remaining)), random_state=seed).copy()

    selected_bg = pd.concat([hard, random], ignore_index=True)
    selected_bg["background_kind"] = ["hard" for _ in range(len(hard))] + ["random" for _ in range(len(random))]

    positives["hard_negative_score"] = np.nan
    positives["background_kind"] = "positive"

    out = pd.concat([positives, selected_bg], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"\n=== {split_name.upper()} ===")
    print(f"All rows before: {len(df)}")
    print(f"Positive candidates retained: {n_pos}")
    print(f"Background available: {len(background)}")
    print(f"Background selected: {len(selected_bg)} | hard={len(hard)} | random={len(random)}")
    print(f"Rows after balance: {len(out)}")
    print("Class counts:")
    print(out["label"].value_counts())
    return out


def main():
    ap = argparse.ArgumentParser(description="Balanced hard-negative sampling for candidate-level OTDR data.")
    ap.add_argument("--input-dir", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--background-per-positive", type=float, default=4.0)
    ap.add_argument("--hard-fraction", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not 0.0 <= args.hard_fraction <= 1.0:
        raise ValueError("--hard-fraction must be in [0, 1]")
    if args.background_per_positive <= 0:
        raise ValueError("--background-per-positive must be > 0")

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / "train_candidates.csv"
    val_path = input_dir / "val_candidates.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing train_candidates.csv or val_candidates.csv. Run build_candidate_train_val.py first.")

    train_raw = pd.read_csv(train_path)
    val_raw = pd.read_csv(val_path)

    train_balanced = balance_one(train_raw, args.background_per_positive,
                                 args.hard_fraction, args.seed, "train")
    val_balanced = balance_one(val_raw, args.background_per_positive,
                               args.hard_fraction, args.seed + 1, "validation")

    train_out = out_dir / "train_candidates_balanced.csv"
    val_out = out_dir / "val_candidates_balanced.csv"
    train_balanced.to_csv(train_out, index=False)
    val_balanced.to_csv(val_out, index=False)

    summary = {
        "input_dir": str(input_dir),
        "background_per_positive": args.background_per_positive,
        "hard_fraction": args.hard_fraction,
        "seed": args.seed,
        "train_rows": len(train_balanced),
        "val_rows": len(val_balanced),
        "train_counts": {str(k): int(v) for k, v in train_balanced["label"].value_counts().to_dict().items()},
        "val_counts": {str(k): int(v) for k, v in val_balanced["label"].value_counts().to_dict().items()},
        "note": "All real candidates retained. Only background candidates were downsampled."
    }
    pd.Series(summary, dtype="object").to_json(out_dir / "balance_summary.json", indent=2, force_ascii=False)

    print(f"\nSaved train -> {train_out}")
    print(f"Saved val   -> {val_out}")
    print(f"Saved summary -> {out_dir / 'balance_summary.json'}")


if __name__ == "__main__":
    main()
