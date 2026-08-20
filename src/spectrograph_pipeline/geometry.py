"""Fiber trace geometry helpers."""

import numpy as np


def evenly_spaced_centers(center: float, spacing: float, n_fibers: int = 7) -> np.ndarray:
    """Return evenly spaced fiber centers symmetric about ``center``."""
    if n_fibers < 1:
        raise ValueError("n_fibers must be at least 1")
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    offsets = np.arange(n_fibers, dtype=float) - (n_fibers - 1) / 2
    return center + spacing * offsets


def parse_centers(value: str) -> np.ndarray:
    """Parse a comma-separated list of fiber centers."""
    try:
        centers = np.asarray([float(item.strip()) for item in value.split(",")], dtype=float)
    except ValueError as exc:
        raise ValueError("trace centers must be comma-separated numbers") from exc

    if centers.size == 0 or not np.all(np.isfinite(centers)):
        raise ValueError("trace centers must contain finite values")
    return centers


def infer_half_width(centers: np.ndarray) -> int:
    """Infer a non-overlapping extraction half-width from fiber separation."""
    centers = np.sort(np.asarray(centers, dtype=float))
    if centers.size < 2:
        raise ValueError("--half-width is required when extracting a single fiber")

    minimum_spacing = np.min(np.diff(centers))
    half_width = int(np.floor(minimum_spacing / 2))
    if half_width < 2:
        raise ValueError("fiber traces are too closely spaced to infer a useful cutout")
    return half_width
