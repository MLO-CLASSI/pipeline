import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import CCDData, VarianceUncertainty

from pipeline.l1.utils import (
    ensure_variance,
    estimate_variance_adu,
    infer_half_width,
    split_bayer,
)


def test_infer_half_width_uses_minimum_spacing_after_sorting():
    centers = np.array([30.0, 10.0, 20.5])

    assert infer_half_width(centers) == 4


def test_infer_half_width_rejects_single_trace():
    with pytest.raises(ValueError, match="single fiber"):
        infer_half_width(np.array([10.0]))


def test_infer_half_width_rejects_too_close_traces():
    with pytest.raises(ValueError, match="too closely spaced"):
        infer_half_width(np.array([10.0, 13.0]))


def test_estimate_variance_adu_combines_shot_and_read_noise():
    data = np.array([-5.0, 0.0, 10.0])

    variance = estimate_variance_adu(data, gain=2.0, read_noise=4.0)

    np.testing.assert_allclose(variance, [4.0, 4.0, 9.0])


@pytest.mark.parametrize("gain", [0.0, -1.0])
def test_estimate_variance_adu_requires_positive_gain(gain):
    with pytest.raises(ValueError, match="gain must be positive"):
        estimate_variance_adu(np.ones(3), gain=gain, read_noise=1.0)


def test_estimate_variance_adu_requires_nonnegative_read_noise():
    with pytest.raises(ValueError, match="read_noise must be non-negative"):
        estimate_variance_adu(np.ones(3), gain=1.0, read_noise=-1.0)


def test_ensure_variance_preserves_existing_uncertainty():
    ccd = CCDData(
        np.ones((2, 2)),
        unit=u.adu,
        uncertainty=VarianceUncertainty(np.full((2, 2), 7.0)),
    )

    result = ensure_variance(ccd, gain=None, read_noise=None)

    assert result is ccd
    np.testing.assert_allclose(result.uncertainty.array, 7.0)


def test_ensure_variance_uses_bias_subtracted_signal_for_poisson_term():
    ccd = CCDData(np.full((2, 2), 110.0), unit=u.adu)
    bias = CCDData(np.full((2, 2), 10.0), unit=u.adu)

    result = ensure_variance(ccd, gain=2.0, read_noise=4.0, bias=bias)

    assert result is not ccd
    np.testing.assert_allclose(result.data, ccd.data)
    np.testing.assert_allclose(result.uncertainty.array, 54.0)


def test_ensure_variance_requires_complete_noise_model():
    ccd = CCDData(np.ones((2, 2)), unit=u.adu)

    with pytest.raises(ValueError, match="provide both --gain and --read-noise"):
        ensure_variance(ccd, gain=1.0, read_noise=None)


def test_split_bayer_masks_nonchannel_pixels_and_preserves_existing_mask():
    existing_mask = np.zeros((4, 4), dtype=bool)
    existing_mask[0, 0] = True
    ccd = CCDData(np.arange(16).reshape(4, 4), unit=u.adu, mask=existing_mask)

    channels = split_bayer(ccd, pattern="RGGB")

    assert set(channels) == {"R", "G1", "G2", "B"}

    expected_unmasked = {
        "R": {(0, 0), (0, 2), (2, 0), (2, 2)},
        "G1": {(0, 1), (0, 3), (2, 1), (2, 3)},
        "G2": {(1, 0), (1, 2), (3, 0), (3, 2)},
        "B": {(1, 1), (1, 3), (3, 1), (3, 3)},
    }

    for name, channel in channels.items():
        assert channel is not ccd
        np.testing.assert_array_equal(channel.data, ccd.data)
        unmasked = set(map(tuple, np.argwhere(~channel.mask)))
        assert unmasked == expected_unmasked[name] - {(0, 0)}
        assert channel.mask[0, 0]


@pytest.mark.parametrize("pattern", ["RGB", "RRBB", "RGGX", ""])
def test_split_bayer_rejects_invalid_patterns(pattern):
    ccd = CCDData(np.zeros((2, 2)), unit=u.adu)

    with pytest.raises(ValueError, match="one R, two Gs, and one B"):
        split_bayer(ccd, pattern=pattern)
