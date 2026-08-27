"""Create normalized JSONL records from the released Track B data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from load_trackb import CORE_FILES, _jsonable, data_root, load_split, training_records, validate_release


TRAINING_SPLITS = ("train.parquet", "val.parquet")
EVALUATION_SPLITS = ("eval_set.parquet", "synthetic_eval_set.parquet", "ood_eval_set.parquet")


def write_jsonl(source: Path, destination: Path) -> int:
    if source.name in TRAINING_SPLITS:
        records = training_records(source.parent, source.name)
    else:
        records = [_jsonable(row) for row in load_split(source.name, source.parent).to_dict(orient="records")]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=CORE_FILES, action="append", help="Split to normalize; repeatable")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    source = data_root(args.data_dir)
    report = validate_release(source)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        return 1
    if args.validate_only:
        return 0

    output = args.output_dir or source / "processed"
    splits = args.split or (*TRAINING_SPLITS, *EVALUATION_SPLITS)
    for split in splits:
        destination = output / f"{split.removesuffix('.parquet')}.jsonl"
        count = write_jsonl(source / split, destination)
        print(f"wrote {count} records: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
