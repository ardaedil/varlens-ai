from __future__ import annotations

import argparse
from pathlib import Path


MODEL_CARD_TEMPLATE = """# VARLens Model Card

## Model Version

{model_version}

## Intended Use

Educational short-clip foul and sanction explanation.

## Unsupported Use

Official refereeing, offside adjudication, handball adjudication, and penalty/no-penalty decisions.

## Dataset Family

SoccerNet-MVFoul, subject to access and redistribution restrictions.

## Required Metrics

- Macro F1
- Balanced accuracy
- Confusion matrices
- Calibration summary
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a VARLens model card scaffold.")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(MODEL_CARD_TEMPLATE.format(model_version=args.model_version))
    print(f"Wrote model card to {args.output}")


if __name__ == "__main__":
    main()
