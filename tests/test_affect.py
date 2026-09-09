import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.affect import (
    CALIBRATION_PROBES,
    EMOTION_LABELS,
    SENTIMENT_PROBES,
    UNRELIABLE_CHANNELS,
    XLM_EMOTION_LABELS,
    derive_ordinal_mapping,
    valence,
)

# --- valence ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ([1.0, 0.0, 0.0, 0.0, 0.0], -1.0),
        ([0.0, 0.0, 0.0, 0.0, 1.0], 1.0),
        ([0.0, 0.0, 1.0, 0.0, 0.0], 0.0),
        ([0.5, 0.0, 0.0, 0.0, 0.5], 0.0),
        ([0.0, 1.0, 0.0, 0.0, 0.0], -0.5),
    ],
)
def test_valence_worked_examples(probabilities, expected):
    assert valence(probabilities) == pytest.approx(expected)


def test_valence_uses_the_expectation_not_the_argmax():
    # Split between the two most negative classes: the argmax would say -1.0,
    # the expectation lands between them.
    assert valence([0.5, 0.5, 0.0, 0.0, 0.0]) == pytest.approx(-0.75)


def test_a_two_class_scale_still_works():
    assert valence([1.0, 0.0]) == -1.0
    assert valence([0.0, 1.0]) == 1.0


def test_one_class_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        valence([1.0])


def test_probabilities_that_do_not_sum_to_one_are_refused():
    with pytest.raises(ValueError, match="sum to"):
        valence([0.5, 0.2, 0.1])


@given(st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=2, max_size=7))
def test_valence_is_always_within_the_scale(raw):
    total = sum(raw)
    assert -1.0 <= valence([x / total for x in raw]) <= 1.0


@given(st.integers(min_value=2, max_value=8))
def test_all_mass_on_an_end_reaches_that_end(n):
    assert valence([1.0] + [0.0] * (n - 1)) == -1.0
    assert valence([0.0] * (n - 1) + [1.0]) == 1.0


# --- ordinal scale direction ------------------------------------------------


def test_a_negative_to_positive_scale_is_detected():
    confusion = {
        "very negative": [0.9, 0.1, 0.0],
        "neutral": [0.1, 0.8, 0.1],
        "very positive": [0.0, 0.1, 0.9],
    }
    assert derive_ordinal_mapping(confusion)[0] is False


def test_a_reversed_scale_is_detected():
    confusion = {
        "very negative": [0.0, 0.1, 0.9],
        "neutral": [0.1, 0.8, 0.1],
        "very positive": [0.9, 0.1, 0.0],
    }
    assert derive_ordinal_mapping(confusion)[0] is True


def test_weak_probes_are_reported():
    confusion = {
        "very negative": [0.3, 0.3, 0.3],
        "very positive": [0.0, 0.1, 0.9],
    }
    assert "very negative" in derive_ordinal_mapping(confusion)[1]


def test_one_class_is_refused_for_the_scale_too():
    with pytest.raises(ValueError, match="at least two"):
        derive_ordinal_mapping({"only": [1.0]})


def test_the_real_calibration_result_reads_as_not_reversed():
    # What NYTK/sentiment-hts5 actually produced: a clean monotonic diagonal.
    confusion = {
        "very negative": [0.76, 0.19, 0.03, 0.01, 0.02],
        "negative": [0.34, 0.58, 0.07, 0.01, 0.01],
        "neutral": [0.05, 0.17, 0.56, 0.18, 0.03],
        "positive": [0.01, 0.01, 0.06, 0.72, 0.21],
        "very positive": [0.01, 0.00, 0.01, 0.06, 0.92],
    }
    reversed_scale, weak = derive_ordinal_mapping(confusion)
    assert reversed_scale is False
    assert weak == ()


# --- the declared constants -------------------------------------------------


def test_every_emotion_label_has_a_probe():
    assert set(EMOTION_LABELS) == set(CALIBRATION_PROBES)


def test_the_multilingual_labels_are_a_subset_of_the_probed_ones():
    assert set(XLM_EMOTION_LABELS) <= set(CALIBRATION_PROBES)


def test_every_probe_set_has_several_texts():
    for label, texts in CALIBRATION_PROBES.items():
        assert len(texts) >= 3, f"{label} needs more than a single probe"


def test_the_sentiment_probes_run_negative_to_positive():
    assert list(SENTIMENT_PROBES)[0] == "very negative"
    assert list(SENTIMENT_PROBES)[-1] == "very positive"


def test_unreliable_channels_name_real_columns():
    # Every entry must be `emotion_<model>_<label>` for a label we score.
    for channel in UNRELIABLE_CHANNELS:
        model, label = channel.removeprefix("emotion_").split("_", 1)
        assert model in {"hu", "xlm"}
        labels = EMOTION_LABELS if model == "hu" else XLM_EMOTION_LABELS
        assert label in labels
