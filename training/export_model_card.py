from __future__ import annotations

import argparse
import json
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

## Training Task

{task}

## Required Metrics

- Macro F1
- Balanced accuracy
- Confusion matrices
- Calibration summary

## Reported Metrics

{reported_metrics}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a VARLens model card scaffold.")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--task", default="sanction")
    parser.add_argument("--metrics-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reported_metrics = "- Pending training run"
    if args.metrics_summary and args.metrics_summary.exists():
        payload = json.loads(args.metrics_summary.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload.get("test_metrics", {}))
        if metrics:
            lines = []
            for key in ("accuracy", "macro_f1", "balanced_accuracy", "macro_precision", "macro_recall"):
                if key in metrics:
                    lines.append(f"- {key}: {metrics[key]}")
            reported_metrics = "\n".join(lines) or reported_metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        MODEL_CARD_TEMPLATE.format(
            model_version=args.model_version,
            task=args.task,
            reported_metrics=reported_metrics,
        ),
        encoding="utf-8",
    )
    print(f"Wrote model card to {args.output}")


if __name__ == "__main__":
    main()
