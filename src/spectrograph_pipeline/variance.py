"""Variance models used by extraction."""

import numpy as np


def estimate_variance_adu(
    data: np.ndarray,
    gain: float,
    read_noise: float,
) -> np.ndarray:
    """Estimate variance in ADU^2 for gain in e-/ADU and read noise in e-."""
    if gain <= 0:
        raise ValueError("gain must be positive")
    if read_noise < 0:
        raise ValueError("read_noise must be non-negative")

    shot_variance = np.clip(np.asarray(data, dtype=float), 0.0, None) / gain
    read_variance = (read_noise / gain) ** 2
    return shot_variance + read_variance
