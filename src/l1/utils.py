"""Detector-domain processing used by the L1 pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.nddata import CCDData, VarianceUncertainty


def infer_half_width(centers: np.ndarray) -> int:
    """Infer a non-overlapping extraction half-width from fiber separation."""
    centers = np.sort(np.asarray(centers, dtype=float))
    if centers.size < 2:
        raise ValueError("Half-width cannot be inferred from single fiber")

    minimum_spacing = np.min(np.diff(centers))
    half_width = int(np.floor(minimum_spacing / 2))
    if half_width < 2:
        raise ValueError("fiber traces are too closely spaced to infer a useful cutout")
    return half_width


def ensure_variance(
    ccd: CCDData,
    gain: float | None,
    read_noise: float | None,
    bias: CCDData | None = None,
) -> CCDData:
    """Ensure an image has variance suitable for optimal extraction.

    When a master bias is available, remove its pedestal before estimating the
    Poisson term. The master calibration image's own uncertainty is not yet
    included in this fallback model.
    """
    if ccd.uncertainty is not None:
        return ccd

    if gain is None or read_noise is None:
        raise ValueError(
            "optimal extraction requires a variance estimate: provide "
            "--variance-ext or both --gain and --read-noise"
        )

    result = ccd.copy()
    signal = result.data if bias is None else result.data - bias.data
    result.uncertainty = VarianceUncertainty(
        estimate_variance_adu(signal, gain, read_noise)
    )
    return result


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
