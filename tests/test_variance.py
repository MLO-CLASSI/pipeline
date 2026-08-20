import numpy as np

from spectrograph_pipeline.variance import estimate_variance_adu


def test_variance_model():
    data = np.array([0.0, 100.0])
    variance = estimate_variance_adu(data, gain=2.0, read_noise=4.0)
    np.testing.assert_allclose(variance, [4.0, 54.0])


def test_negative_signal_does_not_create_negative_variance():
    data = np.array([-10.0])
    variance = estimate_variance_adu(data, gain=2.0, read_noise=4.0)
    np.testing.assert_allclose(variance, [4.0])
