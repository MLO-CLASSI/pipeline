import numpy as np
import pytest

from spectrograph_pipeline.geometry import evenly_spaced_centers, infer_half_width, parse_centers


def test_evenly_spaced_seven_fibers():
    centers = evenly_spaced_centers(100.0, 25.0)
    np.testing.assert_allclose(centers, [25, 50, 75, 100, 125, 150, 175])


def test_parse_centers():
    centers = parse_centers("10.5, 20,30.25")
    np.testing.assert_allclose(centers, [10.5, 20.0, 30.25])


def test_infer_half_width():
    assert infer_half_width(np.array([10.0, 35.7, 61.4])) == 12


def test_single_fiber_requires_explicit_half_width():
    with pytest.raises(ValueError, match="half-width"):
        infer_half_width(np.array([10.0]))
