from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VARLens model evaluation metrics.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text())
    summary = {
        "model_version": predictions.get("model_version", "unknown"),
        "metrics_required": ["macro_f1", "balanced_accuracy", "calibration_summary"],
        "notes": "Populate metrics from validation predictions before release.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote evaluation summary to {args.output}")


if __name__ == "__main__":
    main()
