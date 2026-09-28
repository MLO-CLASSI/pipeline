import numpy as np
import pytest
from astropy import units as u

from pipeline.l1.extraction import boxcar_extract_traces, optimal_extract_traces


@pytest.mark.parametrize("extractor", [boxcar_extract_traces, optimal_extract_traces])
def test_extractors_require_2d_data(extractor):
    with pytest.raises(ValueError, match="expected a 2-D image"):
        extractor(
            np.ones(10),
            np.ones(10),
            centers=[5.0],
            half_width=2,
        )


@pytest.mark.parametrize("extractor", [boxcar_extract_traces, optimal_extract_traces])
def test_extractors_require_matching_variance_shape(extractor):
    with pytest.raises(ValueError, match="variance must have the same shape"):
        extractor(
            np.ones((10, 20)),
            np.ones((9, 20)),
            centers=[5.0],
            half_width=2,
        )


@pytest.mark.parametrize("extractor", [boxcar_extract_traces, optimal_extract_traces])
def test_extractors_require_matching_mask_shape(extractor):
    with pytest.raises(ValueError, match="mask must have the same shape"):
        extractor(
            np.ones((10, 20)),
            np.ones((10, 20)),
            centers=[5.0],
            half_width=2,
            mask=np.zeros((9, 20), dtype=bool),
        )


@pytest.mark.parametrize("extractor", [boxcar_extract_traces, optimal_extract_traces])
def test_extractors_require_minimum_half_width(extractor):
    with pytest.raises(ValueError, match="half_width must be at least 2"):
        extractor(
            np.ones((10, 20)),
            np.ones((10, 20)),
            centers=[5.0],
            half_width=1,
        )


@pytest.mark.parametrize("extractor", [boxcar_extract_traces, optimal_extract_traces])
@pytest.mark.parametrize("center", [0.5, 8.5])
def test_extractors_reject_centers_too_close_to_detector_edge(extractor, center):
    with pytest.raises(ValueError, match="at least one pixel inside"):
        extractor(
            np.ones((10, 20)),
            np.ones((10, 20)),
            centers=[center],
            half_width=2,
        )


def test_boxcar_extracts_synthetic_trace_with_uncertainty():
    ndisp = 40
    center = 10
    profile = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    amplitude = np.linspace(5.0, 15.0, ndisp)
    data = np.zeros((21, ndisp), dtype=float)
    data[center - 2:center + 3, :] = profile[:, None] * amplitude[None, :]
    variance = np.ones_like(data)

    spectrum, = boxcar_extract_traces(
        data,
        variance,
        centers=[center],
        half_width=4,
        unit=u.adu,
    )

    np.testing.assert_allclose(spectrum.flux.value, profile.sum() * amplitude, rtol=1e-10)
    assert spectrum.flux.unit == u.adu
    assert spectrum.uncertainty is not None
    assert np.all(np.isfinite(spectrum.uncertainty.array))
    assert np.all(spectrum.uncertainty.array > 0)
