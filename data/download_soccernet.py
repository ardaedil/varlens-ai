from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SoccerNet-MVFoul download instructions.")
    parser.add_argument("--task", default="mvfouls", choices=["mvfouls"])
    parser.add_argument("--split", default="train", choices=["train", "valid", "test", "challenge"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(
            "Dry run: SoccerNet video data requires authorized access. "
            f"Requested task={args.task} split={args.split}."
        )
        return

    raise SystemExit(
        "SoccerNet-MVFoul downloads are gated by SoccerNet access terms. "
        "Validate credentials and redistribution rights before automating this step."
    )


if __name__ == "__main__":
    main()
