#!/usr/bin/env python3
"""Download the external skill metadata and retrieval index for Track B.

The Track B parquet release does not include the full retrieval corpus. This
script downloads only the artifacts required by trackb_cons_anchor_example.py:

- skillusage/skills/skills_meta.jsonl
- skillusage/search_server/index/ (including skills.db)

Usage:
    python scripts/release/download_trackb_index.py
    python scripts/release/download_trackb_index.py --force
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

HF_REPO = "Shiyu-Lab/Skill-Usage"
HF_SKILLS_META = "skills-34k/skills_meta.jsonl"
HF_INDEX_ARCHIVE = "search_index/search_index.zip"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKILLS_META = REPO_ROOT / "data" / "skills_meta.jsonl"
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"
DEFAULT_CACHE_DIR = REPO_ROOT / ".hf_download"


def download_file(filename: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=filename,
        local_dir=str(cache_dir),
    ))


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_root):
                raise ValueError(f"archive member escapes destination: {member.filename}")
        archive.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-meta", type=Path, default=DEFAULT_SKILLS_META)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--force", action="store_true",
                        help="Download and extract again even if outputs already exist.")
    args = parser.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: missing dependency. Install it with:", file=sys.stderr)
        print("  pip install huggingface_hub", file=sys.stderr)
        return 1

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    index_db = args.index_dir / "skills.db"

    if args.force or not args.skills_meta.is_file():
        print(f"[download] {HF_SKILLS_META}")
        source_meta = download_file(HF_SKILLS_META, args.cache_dir)
        args.skills_meta.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_meta, args.skills_meta)
        print(f"[ok] skill metadata -> {args.skills_meta}")
    else:
        print(f"[skip] skill metadata exists: {args.skills_meta}")

    if args.force or not index_db.is_file():
        print(f"[download] {HF_INDEX_ARCHIVE}")
        archive_path = download_file(HF_INDEX_ARCHIVE, args.cache_dir)
        extract_archive(archive_path, args.index_dir)
        print(f"[ok] retrieval index -> {args.index_dir}")
    else:
        print(f"[skip] retrieval index exists: {index_db}")

    missing = [str(path) for path in (args.skills_meta, index_db) if not path.is_file()]
    if missing:
        print(f"ERROR: required output missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    print("[ok] Track B retrieval artifacts are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
