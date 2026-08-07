import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.cap import (
    DEFAULT_THRESHOLD,
    MIX_LABEL,
    aggregate_scores,
    apply_threshold,
)

# --- the confidence rule ----------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (1.0, "Health"),
        (0.61, "Health"),
        (0.60, "Health"),  # the model card's rule is >= 0.60
        (0.599, MIX_LABEL),
        (0.0, MIX_LABEL),
    ],
)
def test_apply_threshold_boundary(score, expected):
    assert apply_threshold("Health", score) == expected


def test_other_is_a_real_label_not_mix():
    """`Other` is a confident judgement; `Mix` is our uncertainty override."""
    assert apply_threshold("Other", 0.95) == "Other"
    assert apply_threshold("Other", 0.10) == MIX_LABEL


def test_apply_threshold_is_configurable():
    assert apply_threshold("Health", 0.5, threshold=0.4) == "Health"


@pytest.mark.parametrize("score", [-0.01, 1.01, 12.0])
def test_apply_threshold_rejects_non_probabilities(score):
    """A value outside [0, 1] means the caller passed a logit."""
    with pytest.raises(ValueError, match="probability"):
        apply_threshold("Health", score)


@given(score=st.floats(min_value=0.0, max_value=1.0))
def test_apply_threshold_only_ever_returns_label_or_mix(score):
    assert apply_threshold("Labor", score) in {"Labor", MIX_LABEL}


@given(score=st.floats(min_value=0.0, max_value=1.0))
def test_threshold_is_monotone(score):
    """Raising the threshold can only ever turn a label into Mix."""
    low = apply_threshold("Labor", score, threshold=0.3)
    high = apply_threshold("Labor", score, threshold=0.9)
    assert not (low == MIX_LABEL and high == "Labor")


def test_default_threshold_matches_the_model_card():
    assert DEFAULT_THRESHOLD == 0.60


# --- aggregation ------------------------------------------------------------


def test_aggregate_averages_distributions():
    label, score, dist = aggregate_scores(
        [{"Health": 0.9, "Labor": 0.1}, {"Health": 0.3, "Labor": 0.7}]
    )
    assert label == "Health"
    assert score == pytest.approx(0.6)
    assert dist["Labor"] == pytest.approx(0.4)


def test_aggregate_single_window_is_identity():
    assert aggregate_scores([{"Health": 0.8, "Labor": 0.2}])[:2] == ("Health", 0.8)


def test_aggregate_weights_by_window_length():
    unweighted, _, _ = aggregate_scores(
        [{"Health": 0.9, "Labor": 0.1}, {"Health": 0.3, "Labor": 0.7}]
    )
    weighted, _, _ = aggregate_scores(
        [{"Health": 0.9, "Labor": 0.1}, {"Health": 0.3, "Labor": 0.7}],
        weights=[10, 250],
    )
    assert unweighted == "Health"
    assert weighted == "Labor"


def test_aggregate_prefers_confidence_over_headcount():
    """Averaging distributions, not voting: one certain window can outweigh two."""
    label, _, _ = aggregate_scores(
        [
            {"Health": 0.99, "Labor": 0.01},
            {"Health": 0.45, "Labor": 0.55},
            {"Health": 0.45, "Labor": 0.55},
        ]
    )
    assert label == "Health"


def test_aggregate_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        aggregate_scores([])


def test_aggregate_mismatched_labels_raise():
    with pytest.raises(ValueError, match="window 1"):
        aggregate_scores([{"Health": 1.0}, {"Labor": 1.0}])


def test_aggregate_wrong_weight_count_raises():
    with pytest.raises(ValueError, match="2 weights for 1 windows"):
        aggregate_scores([{"Health": 1.0}], weights=[1, 2])


def test_aggregate_zero_weights_raise():
    with pytest.raises(ValueError, match="positive"):
        aggregate_scores([{"Health": 1.0}], weights=[0])


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=12,
    )
)
def test_aggregated_distribution_stays_a_distribution(health_probs):
    """Averaging probability distributions yields a probability distribution."""
    windows = [{"Health": p, "Labor": 1.0 - p} for p in health_probs]
    _, score, dist = aggregate_scores(windows)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in dist.values())
    assert score == max(dist.values())
