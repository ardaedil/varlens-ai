from training.train_videomae import compute_class_weights


def test_effective_number_weights_are_mean_one_and_favor_rare_classes():
    weights = compute_class_weights(
        {"common": 100, "rare": 10},
        strategy="effective_num",
        beta=0.9999,
    )

    assert weights[1] > weights[0]
    assert round(sum(weights) / len(weights), 6) == 1.0


def test_none_class_weighting_returns_uniform_weights():
    assert compute_class_weights(
        {"a": 100, "b": 1},
        strategy="none",
    ) == [1.0, 1.0]
