from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.manifest import build_manifest_document, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-view MVFoul training manifest.")
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--default-split")
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    document = build_manifest_document(
        annotation_paths=args.annotations,
        video_root=args.video_root,
        default_split=args.default_split,
        verify_files=args.verify_files,
    )
    write_manifest(document, args.output)

    summary = document["summary"]
    print(
        "Wrote "
        f"{summary['record_count']} view records across {summary['action_count']} actions "
        f"to {args.output}"
    )
    for split, counts in summary["split_counts"].items():
        print(f"  {split}: {counts['records']} views / {counts['actions']} actions")


if __name__ == "__main__":
    main()
