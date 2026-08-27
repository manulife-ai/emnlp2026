#!/usr/bin/env python3
"""Build task-skill records and task-disjoint splits from normalized trials.

This script is intentionally independent of the rest of the repository. It
creates pair tables suitable for a future skill-router training pipeline and
keeps all trials for a task in the same split.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def _split_tasks(tasks: list[str], seed: int, train_fraction: float, val_fraction: float) -> dict[str, list[str]]:
    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    return {
        "train": sorted(shuffled[:n_train]),
        "validation": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }


def build_pairs(frame: pd.DataFrame, positive_threshold: float) -> pd.DataFrame:
    has_skills = frame["injected_skills_list"].map(lambda value: value is not None and len(value) > 0)
    clean = frame[frame["is_clean_scored"] & has_skills].copy()
    exploded = clean[["benchmark", "task_name", "injected_skills_list", "effective_reward"]].explode(
        "injected_skills_list", ignore_index=True
    )
    exploded = exploded.rename(columns={"injected_skills_list": "skill_id"})
    pairs = (
        exploded.groupby(["benchmark", "task_name", "skill_id"], as_index=False)
        .agg(mean_effective_reward=("effective_reward", "mean"), n_trials=("effective_reward", "size"))
    )
    pairs["task_id"] = pairs["benchmark"] + ":" + pairs["task_name"]
    pairs["is_positive"] = pairs["mean_effective_reward"] > positive_threshold
    pairs["label"] = pairs["is_positive"].astype("int8")
    return pairs.sort_values(["benchmark", "task_name", "skill_id"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Normalized parquet from preprocess_trials.py")
    parser.add_argument("output_dir", type=Path, help="Directory for pair tables and split metadata")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--include-terminalbench-train",
        action="store_true",
        help="Allow Terminal-Bench tasks in train/validation; default keeps them test-only.",
    )
    args = parser.parse_args()
    if args.train_fraction + args.validation_fraction >= 1:
        raise ValueError("train fraction + validation fraction must be less than 1")

    pairs = build_pairs(pd.read_parquet(args.input), args.positive_threshold)
    split: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for benchmark, group in pairs.groupby("benchmark"):
        tasks = sorted(group["task_name"].unique())
        if benchmark == "terminalbench" and not args.include_terminalbench_train:
            split["test"].extend(f"{benchmark}:{task}" for task in tasks)
        else:
            local = _split_tasks(tasks, args.seed, args.train_fraction, args.validation_fraction)
            for name in split:
                split[name].extend(f"{benchmark}:{task}" for task in local[name])
    for name in split:
        split[name] = sorted(set(split[name]))

    task_to_split = {
        (benchmark, task): name
        for name, task_ids in split.items()
        for benchmark, task in (task_id.split(":", 1) for task_id in task_ids)
    }
    pair_split = pairs.apply(
        lambda row: task_to_split[(row["benchmark"], row["task_name"])], axis=1
    )
    pairs["split"] = pair_split

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        pairs[pairs["split"] == name].to_parquet(args.output_dir / f"{name}_pairs.parquet", index=False)
    metadata = {
        "seed": args.seed,
        "positive_threshold": args.positive_threshold,
        "split_unit": "benchmark-qualified task_name; all trials for a task stay together",
        "terminalbench_policy": "test_only" if not args.include_terminalbench_train else "same_as_skillsbench",
        "tasks_by_split": split,
        "rows_by_split": pairs["split"].value_counts().to_dict(),
        "pairs_by_label": pairs["label"].value_counts().to_dict(),
    }
    (args.output_dir / "task_split.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pairs.to_parquet(args.output_dir / "all_pairs.parquet", index=False)
    print(f"Wrote {len(pairs):,} aggregated task-skill pairs to {args.output_dir}")
    print(f"Tasks: train={len(split['train'])}, validation={len(split['validation'])}, test={len(split['test'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
