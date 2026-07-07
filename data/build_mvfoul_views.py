from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-view MVFoul training manifest.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text())
    records = []
    for action in annotations.get("actions", []):
        action_id = action["id"]
        for view in action.get("views", []):
            records.append(
                {
                    "action_id": action_id,
                    "view_id": view["id"],
                    "video_path": view["video_path"],
                    "sanction_label": action["sanction_label"],
                    "action_type_label": action.get("action_type_label", "unknown"),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote {len(records)} view records to {args.output}")


if __name__ == "__main__":
    main()
