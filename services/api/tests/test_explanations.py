from packages.contracts.analyze import ActionTypePrediction, SanctionPrediction
from services.api.app.explanations.templates import build_explanation


def test_low_confidence_explanation_stays_cautious():
    explanation, focus, limitations = build_explanation(
        SanctionPrediction(
            label="offence_yellow",
            confidence=0.63,
            alternatives=[{"label": "offence_no_card", "confidence": 0.24}],
        ),
        ActionTypePrediction(
            label="holding",
            confidence=0.58,
            alternatives=[{"label": "pushing", "confidence": 0.21}],
        ),
    )

    assert "not a definitive decision" in explanation
    assert "referee remains the final authority" in explanation
    assert any("Confidence is moderate" in item for item in limitations)
    assert any("grabs" in item for item in focus)


def test_high_confidence_explanation_keeps_single_view_limitation():
    _, _, limitations = build_explanation(
        SanctionPrediction(label="offence_red", confidence=0.86, alternatives=[]),
        ActionTypePrediction(label="elbowing", confidence=0.83, alternatives=[]),
    )

    assert "Single-view input can miss decisive angles." in limitations
