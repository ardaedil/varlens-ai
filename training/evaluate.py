from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.metrics import compute_classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VARLens model evaluation metrics.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    references = predictions["references"]
    predicted = predictions["predictions"]
    id2label = {int(key): value for key, value in predictions.get("id2label", {}).items()}
    summary = {
        "model_version": predictions.get("model_version", "unknown"),
        "metrics": compute_classification_metrics(references, predicted, id2label=id2label or None),
        "metrics_required": ["macro_f1", "balanced_accuracy", "confusion_matrix"],
        "notes": "Add calibration summaries once confidence calibration is part of the training flow.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote evaluation summary to {args.output}")


if __name__ == "__main__":
    main()
