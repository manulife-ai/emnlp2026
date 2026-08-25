#!/usr/bin/env python3
"""Validate and normalize a released Harbor trial parquet file.

The output is self-contained and adds analysis-friendly columns without
requiring the rest of the skill-router repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


JSON_LIST_COLUMNS = {
    "injected_skills": "injected_skills_list",
    "skill_invocation_args": "skill_invocation_args_list",
}
JSON_DICT_COLUMNS = {
    "tool_call_counts": "tool_call_counts_map",
}


def _decode(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON value: {value!r}") from exc
    if not isinstance(parsed, type(default)):
        raise ValueError(f"Expected {type(default).__name__}, got {type(parsed).__name__}")
    return parsed


def normalize_trials(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"task_name", "trial_name", "benchmark", "reward", "soft_reward", "injected_skills"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    output = frame.copy()
    for source, target in JSON_LIST_COLUMNS.items():
        output[target] = output[source].map(lambda value: _decode(value, []))
    for source, target in JSON_DICT_COLUMNS.items():
        output[target] = output[source].map(lambda value: _decode(value, {}))

    output["effective_reward"] = pd.to_numeric(output["soft_reward"], errors="coerce").fillna(
        pd.to_numeric(output["reward"], errors="coerce")
    )
    output["benchmark_task_id"] = output["benchmark"].astype(str) + ":" + output["task_name"].astype(str)
    output["injected_skill_count"] = output["injected_skills_list"].map(len)
    output["is_clean_scored"] = output["exception_type"].isna() & output["effective_reward"].notna()
    output["is_positive_candidate"] = output["is_clean_scored"] & (output["effective_reward"] > 0.5)

    duplicate_keys = output.duplicated(["benchmark", "task_name", "trial_name"]).sum()
    if duplicate_keys:
        raise ValueError(f"Found {duplicate_keys} duplicate benchmark/task/trial keys")
    invalid_soft = output["soft_reward"].notna() & ~output["soft_reward"].between(0, 1)
    if invalid_soft.any():
        raise ValueError(f"Found {int(invalid_soft.sum())} soft_reward values outside [0, 1]")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input Harbor trial parquet")
    parser.add_argument("output", type=Path, help="Normalized output parquet")
    args = parser.parse_args()

    frame = normalize_trials(pd.read_parquet(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"Wrote {len(frame):,} rows x {len(frame.columns)} columns to {args.output}")
    print(f"Clean scored rows: {int(frame['is_clean_scored'].sum()):,}")
    print(f"Positive candidates (>0.5 effective_reward): {int(frame['is_positive_candidate'].sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
