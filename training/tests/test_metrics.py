from training.metrics import compute_classification_metrics


def test_compute_classification_metrics_reports_macro_scores():
    metrics = compute_classification_metrics(
        references=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        id2label={0: "no_offence", 1: "offence_yellow"},
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["macro_recall"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["per_class"]["no_offence"]["support"] == 2
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
