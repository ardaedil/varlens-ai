from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VideoMAE classifier for VARLens MVFoul v1.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    if args.dry_run:
        print(f"Dry run ok: would train from {args.manifest} into {args.output_dir}")
        return

    raise SystemExit(
        "Training requires transformers, PyTorch, video decoding dependencies, and authorized MVFoul data. "
        "Use Hugging Face VideoMAE fine-tuning with the manifest produced by data/build_mvfoul_views.py."
    )


if __name__ == "__main__":
    main()
