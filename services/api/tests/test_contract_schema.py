import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]


def test_golden_output_matches_json_schema():
    schema = json.loads((ROOT / "packages/contracts/analyze.schema.json").read_text())
    fixture = json.loads((ROOT / "tests/fixtures/golden_outputs/ok_holding.json").read_text())

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(fixture), key=lambda error: error.path)

    assert errors == []


def test_schema_forbids_official_decision_claims():
    schema = json.loads((ROOT / "packages/contracts/analyze.schema.json").read_text())
    assert schema["$defs"]["reviewContext"]["properties"]["official_decision_claimed"]["const"] is False
