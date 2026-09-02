"""Optimal extraction operations for L1 products."""

from collections.abc import Sequence

import numpy as np
from astropy import units as u
from astropy.modeling import models
from astropy.nddata import CCDData, VarianceUncertainty
from specreduce.extract import BoxcarExtract, HorneExtract
from specreduce.tracing import ArrayTrace, FitTrace, FlatTrace


def boxcar_extract_traces(
    data: np.ndarray,
    variance: np.ndarray,
    centers: Sequence[float],
    half_width: int,
    mask: np.ndarray | None = None,
    unit: u.UnitBase = u.adu,
):
    """Naive boxcar extraction"""
    data = np.asarray(data, dtype=float)
    variance = np.asarray(variance, dtype=float)

    if data.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {data.shape}")
    if variance.shape != data.shape:
        raise ValueError("variance must have the same shape as data")
    if mask is not None and np.shape(mask) != data.shape:
        raise ValueError("mask must have the same shape as data")
    if half_width < 2:
        raise ValueError("half_width must be at least 2 pixels")

    spectra = []
    for center in centers:
        if center < 1 or center > data.shape[0] - 2:
            raise ValueError(
                f"trace center {center} must be at least one pixel inside "
                f"cross-dispersion bounds 0..{data.shape[0] - 1}"
            )

        lower = max(0, int(np.floor(center)) - half_width)
        upper = min(data.shape[0], int(np.floor(center)) + half_width + 1)
        local_center = float(center - lower)

        cutout = data[lower:upper, :]
        variance_cutout = variance[lower:upper, :]
        mask_cutout = None if mask is None else mask[lower:upper, :]

        if local_center < 1 or local_center > cutout.shape[0] - 2:
            raise ValueError(
                f"trace center {center} is too close to an image edge for "
                f"half-width {half_width}"
            )

        trace_img = CCDData(cutout, unit=unit, mask=mask_cutout,
                            uncertainty=VarianceUncertainty(variance_cutout))
        trace = FitTrace(trace_img,
                         window=None,
                         bins=10, guess=local_center,
                         peak_method="centroid")
        # trace = FlatTrace(cutout, trace_pos=local_center)
        # trace = ArrayTrace(cutout, trace=)
        zero_background = models.Const1D(0.)
        zero_background.amplitude.fixed = True
        extraction = BoxcarExtract(
            trace_img,
            trace,
            width=2*half_width,
            disp_axis=1,
            crossdisp_axis=0,
        )
        spectra.append(extraction.spectrum)

    return spectra

def optimal_extract_traces(
    data: np.ndarray,
    variance: np.ndarray,
    centers: Sequence[float],
    half_width: int,
    mask: np.ndarray | None = None,
    unit: u.UnitBase = u.adu,
):
    """Horne-extract flat traces from independent local cutouts.

    The image is assumed to have dispersion along axis 1 and cross-dispersion
    along axis 0. Background subtraction is assumed to have been performed
    upstream or to be negligible; the fitted Horne spatial profile therefore
    uses a fixed zero background term.
    """
    data = np.asarray(data, dtype=float)
    variance = np.asarray(variance, dtype=float)

    if data.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {data.shape}")
    if variance.shape != data.shape:
        raise ValueError("variance must have the same shape as data")
    if mask is not None and np.shape(mask) != data.shape:
        raise ValueError("mask must have the same shape as data")
    if half_width < 2:
        raise ValueError("half_width must be at least 2 pixels")

    spectra = []
    for center in centers:
        if center < 1 or center > data.shape[0] - 2:
            raise ValueError(
                f"trace center {center} must be at least one pixel inside "
                f"cross-dispersion bounds 0..{data.shape[0] - 1}"
            )

        lower = max(0, int(np.floor(center)) - half_width)
        upper = min(data.shape[0], int(np.floor(center)) + half_width + 1)
        local_center = float(center - lower)

        cutout = data[lower:upper, :]
        variance_cutout = variance[lower:upper, :]
        mask_cutout = None if mask is None else mask[lower:upper, :]

        if local_center < 1 or local_center > cutout.shape[0] - 2:
            raise ValueError(
                f"trace center {center} is too close to an image edge for "
                f"half-width {half_width}"
            )

        trace_img = CCDData(cutout, unit=unit, mask=mask_cutout)
        # trace = FlatTrace(cutout, trace_pos=local_center)
        trace = FitTrace(trace_img,
                         window=None,
                         bins=10, guess=local_center,
                         peak_method="centroid")
        zero_background = models.Const1D(0.)
        zero_background.amplitude.fixed = True
        extraction = HorneExtract(
            cutout,
            trace,
            variance=variance_cutout,
            mask=mask_cutout,
            unit=unit,
            disp_axis=1,
            crossdisp_axis=0,
            spatial_profile="gaussian",
            bkgrd_prof=zero_background,
        )
        spectra.append(extraction.spectrum)

    return spectra
