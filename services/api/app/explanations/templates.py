from __future__ import annotations

from packages.contracts.analyze import ActionTypePrediction, SanctionPrediction

SANCTION_TEXT = {
    "no_offence": "The model does not see enough in this clip to call an offence from the available view.",
    "offence_no_card": "The model sees a foul-like offence pattern, but not a card-level pattern.",
    "offence_yellow": "The model leans toward a caution-level offence pattern.",
    "offence_red": "The model sees a potentially serious offence pattern that a referee-style review would inspect closely.",
}

ACTION_TEXT = {
    "standing_tackle": "The key visual question is whether the standing challenge makes unfair contact before or through the ball.",
    "tackle": "The key visual question is whether the tackle wins the ball cleanly or trips or impedes the opponent.",
    "holding": "The key visual question is whether a player grabs or restrains an opponent enough to impede movement.",
    "pushing": "The key visual question is whether arm or body contact clearly displaces the opponent.",
    "challenge": "The key visual question is whether the challenge is careless, reckless, or uses excessive force.",
    "dive": "The key visual question is whether the player initiates or exaggerates contact.",
    "high_leg": "The key visual question is whether the raised leg creates danger or makes unfair contact.",
    "elbowing": "The key visual question is whether elbow or forearm contact is deliberate or reckless.",
    "unknown": "The action type is unclear from the available view.",
}

FOCUS_BY_ACTION = {
    "standing_tackle": [
        "Check whether the defender plays the ball before making contact.",
        "Check the contact point and whether studs are exposed.",
    ],
    "tackle": [
        "Check whether the tackler wins the ball cleanly.",
        "Check whether the follow-through trips or endangers the opponent.",
    ],
    "holding": [
        "Check whether the defender grabs and impedes the attacker.",
        "Check whether the attacker keeps control of the ball after contact.",
    ],
    "pushing": [
        "Check whether arm or body contact displaces the opponent.",
        "Check whether the opponent had a realistic chance to continue the play.",
    ],
    "challenge": [
        "Check the speed, contact point, and force of the challenge.",
        "Check whether the player had a fair chance to play the ball.",
    ],
    "dive": [
        "Check whether contact is visible before the player goes down.",
        "Check whether the player initiates or exaggerates the fall.",
    ],
    "high_leg": [
        "Check whether the raised leg endangers an opponent.",
        "Check whether the opponent pulls away to avoid contact.",
    ],
    "elbowing": [
        "Check whether elbow or forearm contact is visible.",
        "Check whether the arm movement appears deliberate or reckless.",
    ],
    "unknown": [
        "Check whether the main angle hides the initial point of contact.",
        "Check whether another view would clarify the action type.",
    ],
}


def build_explanation(
    sanction_prediction: SanctionPrediction,
    action_prediction: ActionTypePrediction,
) -> tuple[str, list[str], list[str]]:
    sanction_text = SANCTION_TEXT[sanction_prediction.label]
    action_text = ACTION_TEXT[action_prediction.label]
    uncertainty_text = ""
    if sanction_prediction.confidence < 0.7 or action_prediction.confidence < 0.7:
        uncertainty_text = " Confidence is not high, so nearby classes should remain in view."

    explanation = (
        f"{sanction_text} {action_text}{uncertainty_text} "
        "This is not a definitive decision, and the referee remains the final authority."
    )

    focus = [
        *FOCUS_BY_ACTION[action_prediction.label],
        "Check whether the available angle hides decisive contact.",
    ]

    limitations = [
        "Single-view input can miss decisive angles.",
        "This system is educational and does not replace the referee.",
    ]
    if sanction_prediction.confidence < 0.7 or action_prediction.confidence < 0.7:
        limitations.append("Confidence is moderate, so nearby classes remain plausible.")

    return explanation, focus, limitations
