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


def split_bayer(ccd: CCDData, pattern: str = "RGGB") -> dict[str, CCDData]:
    """Split a Bayer CFA image into R/G1/G2/B masked CCDData images.

    Non-channel pixels are masked rather than interpolated, so all outputs
    retain the original detector dimensions and pixel coordinates.

    ``pattern`` gives the 2x2 Bayer pattern in row-major order, e.g. "RGGB"
    """
    if len(pattern) != 4 or sorted(pattern.upper()) != ["B", "G", "G", "R"]:
        raise ValueError("pattern must contain one R, two Gs, and one B")

    pattern = pattern.upper()
    labels = np.array(list(pattern), dtype="U2").reshape(2, 2)

    green = np.argwhere(labels == "G")
    labels[tuple(green[0])] = "G1"
    labels[tuple(green[1])] = "G2"

    y, x = np.indices(ccd.shape)
    cfa = labels[y % 2, x % 2]

    existing_mask = np.zeros(ccd.shape, dtype=bool) if ccd.mask is None else np.asarray(ccd.mask, dtype=bool)

    channels = {}

    for name in ("R", "G1", "G2", "B"):
        channel = ccd.copy()
        channel.mask = existing_mask | (cfa != name)
        channels[name] = channel

    return channels
