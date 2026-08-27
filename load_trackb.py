"""Load and validate the released Track B parquet files.

This module intentionally has no dependency on the source repository. It is
usable from a downloaded copy of this data directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

CORE_FILES = (
    "positives.parquet",
    "negatives.parquet",
    "task_positive_sets.parquet",
    "train.parquet",
    "val.parquet",
    "eval_set.parquet",
    "synthetic_eval_set.parquet",
    "ood_eval_set.parquet",
)

EXPECTED_ROWS = {
    "positives.parquet": 255,
    "negatives.parquet": 1239,
    "task_positive_sets.parquet": 68,
    "train.parquet": 13271,
    "val.parquet": 556,
    "eval_set.parquet": 78,
    "synthetic_eval_set.parquet": 4023,
    "ood_eval_set.parquet": 15,
}

TRAIN_REQUIRED_COLUMNS = {
    "query_id",
    "positive_set_id",
    "query_text",
    "positive_skill_id",
    "positive_skill_text",
    "positive_weight",
    "negatives",
}

EVAL_REQUIRED_COLUMNS = {
    "query_id",
    "query_text",
    "positive_skill_id",
    "positive_set_id",
}


def data_root(path: str | Path | None = None) -> Path:
    """Return a Track B data directory, defaulting to this script's directory."""
    return Path(path).expanduser().resolve() if path else Path(__file__).resolve().parent


def load_split(name: str, root: str | Path | None = None) -> pd.DataFrame:
    """Load one released parquet split without modifying its values."""
    path = data_root(root) / name
    if path.suffix != ".parquet":
        raise ValueError(f"Expected a parquet filename, got: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _missing(columns: set[str], required: set[str]) -> list[str]:
    return sorted(required - columns)


def validate_release(root: str | Path | None = None) -> dict[str, Any]:
    """Validate required files, row counts, and model-facing columns."""
    root_path = data_root(root)
    report: dict[str, Any] = {"root": str(root_path), "files": {}, "valid": True}
    for name in CORE_FILES:
        path = root_path / name
        item: dict[str, Any] = {"exists": path.is_file()}
        if path.is_file():
            frame = pd.read_parquet(path)
            item["rows"] = len(frame)
            item["expected_rows"] = EXPECTED_ROWS[name]
            item["row_count_matches"] = len(frame) == EXPECTED_ROWS[name]
            if name in {"train.parquet", "val.parquet"}:
                item["missing_columns"] = _missing(set(frame.columns), TRAIN_REQUIRED_COLUMNS)
            elif name.endswith("eval_set.parquet"):
                item["missing_columns"] = _missing(set(frame.columns), EVAL_REQUIRED_COLUMNS)
            else:
                item["missing_columns"] = []
        report["files"][name] = item
        report["valid"] &= item["exists"] and item.get("row_count_matches", False)
        report["valid"] &= not item.get("missing_columns", [])
    return report


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if pd.isna(value):
        return None
    return value


def training_records(root: str | Path | None = None, split: str = "train.parquet") -> list[dict[str, Any]]:
    """Return normalized contrastive records for a training-style split."""
    frame = load_split(split, root)
    missing = _missing(set(frame.columns), TRAIN_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"{split} is missing required columns: {', '.join(missing)}")
    records = []
    for row in frame.to_dict(orient="records"):
        negatives = row.get("negatives")
        if negatives is None:
            negatives = []
        row["negatives"] = _jsonable(negatives)
        records.append(_jsonable(row))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print the validation report as JSON")
    args = parser.parse_args()
    report = validate_release(args.data_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name, item in report["files"].items():
            print(f"{name}: rows={item.get('rows', 0)} valid={item.get('exists') and item.get('row_count_matches') and not item.get('missing_columns')}")
        print(f"release_valid={report['valid']}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
