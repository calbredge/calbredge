"""How a conversion is scored (agent/calbredge/validate_rknn.py): float and quantized outputs
paired by shape, then each channel band compared with itself.

These run on arrays alone. The checks against the real toolkit are
tests/open/test_validate_rknn.py, which needs rknn-toolkit2.
"""

import numpy as np
import pytest
from calbredge.validate_rknn import (
    BAND_SIZE,
    COLLAPSE_RATIO,
    MEANINGFUL_FLOOR,
    CollapsedBand,
    _align_outputs_by_shape,
    _as_channel_major,
    _channel_bands,
    _score_aligned_outputs,
)

pytestmark = pytest.mark.unit


def _score(float_outputs, quant_outputs):
    pairs, reason = _align_outputs_by_shape(float_outputs, quant_outputs)
    assert reason is None
    assert pairs is not None
    return _score_aligned_outputs(pairs, BAND_SIZE, MEANINGFUL_FLOOR, COLLAPSE_RATIO)


def test_outputs_are_paired_by_shape_whatever_their_order():
    boxes, scores = np.ones((1, 4, 10)), np.full((1, 80, 10), 2.0)
    pairs, reason = _align_outputs_by_shape([boxes, scores], [scores * 3, boxes * 3])
    assert reason is None
    assert pairs is not None
    assert [(i, f.shape, q.shape) for i, f, q in pairs] == [
        (0, (1, 4, 10), (1, 4, 10)),
        (1, (1, 80, 10), (1, 80, 10)),
    ]
    assert float(pairs[0][2].max()) == 3.0
    assert float(pairs[1][2].max()) == 6.0


def test_two_outputs_of_one_shape_pair_in_order():
    first, second = np.zeros((1, 8)), np.ones((1, 8))
    pairs, _ = _align_outputs_by_shape([first, second], [first + 5, second + 5])
    assert pairs is not None
    assert [float(q.max()) for _, _, q in pairs] == [5.0, 6.0]


def test_a_different_number_of_outputs_is_a_failure_with_its_reason():
    pairs, reason = _align_outputs_by_shape([np.ones((1, 8))], [])
    assert pairs is None
    assert reason == (
        "output count mismatch: 1 float reference output(s) vs 0 quantized simulator output(s)"
    )


def test_an_output_whose_shape_changed_is_a_failure_with_its_reason():
    pairs, reason = _align_outputs_by_shape([np.ones((1, 84, 10))], [np.ones((1, 10, 84))])
    assert pairs is None
    assert reason == (
        "no quantized output with shape (1, 84, 10) to match float reference output #0 "
        "-- conversion changed output shape/structure"
    )


def test_channels_are_cut_into_bands_with_the_remainder_last():
    assert _channel_bands(16, 8) == [(0, 8), (8, 16)]
    assert _channel_bands(20, 8) == [(0, 8), (8, 16), (16, 20)]
    assert _channel_bands(3, 8) == [(0, 3)]


def test_an_output_with_no_batch_axis_gets_one():
    assert _as_channel_major(np.arange(5.0)).shape == (1, 5)
    kept = np.zeros((1, 5, 2))
    assert _as_channel_major(kept) is kept


def test_a_faithful_conversion_passes_and_counts_its_bands():
    reference = np.random.default_rng(0).uniform(0.5, 1.0, size=(1, 20, 6))
    result = _score([reference], [reference * 0.97])
    assert result.passed
    assert result.verdict == "pass"
    assert result.collapsed_bands == []
    assert (result.meaningful_bands, result.total_bands) == (3, 3)
    assert result.reason is None


def test_a_band_whose_signal_died_fails_and_is_named():
    reference = np.full((1, 16, 4), 0.8)
    quantized = reference.copy()
    quantized[:, 8:16] = 0.01  # under 5% of 0.8: the class-score band went dead
    result = _score([np.ones((1, 4)), reference], [np.ones((1, 4)), quantized])
    assert not result.passed
    assert result.verdict == "fail"
    assert result.collapsed_bands == [
        CollapsedBand(output_index=1, channel_range=(8, 16), float_energy=0.8, quant_energy=0.01)
    ]
    assert (result.meaningful_bands, result.total_bands) == (3, 3)


def test_the_sign_of_a_band_does_not_matter():
    reference = np.full((1, 8), -0.9)
    assert _score([reference], [reference * 0.5]).passed
    assert not _score([reference], [np.zeros((1, 8))]).passed


def test_a_band_that_was_quiet_before_is_never_a_collapse():
    """Nothing to lose: the float band's own energy is at or under the meaningful floor."""
    reference = np.full((1, 8), MEANINGFUL_FLOOR)
    result = _score([reference], [np.zeros((1, 8))])
    assert result.passed
    assert (result.meaningful_bands, result.total_bands) == (0, 1)


def test_exactly_the_collapse_ratio_is_not_a_collapse():
    reference = np.full((1, 8), 1.0)
    assert _score([reference], [np.full((1, 8), COLLAPSE_RATIO)]).passed
    assert not _score([reference], [np.full((1, 8), COLLAPSE_RATIO * 0.99)]).passed
