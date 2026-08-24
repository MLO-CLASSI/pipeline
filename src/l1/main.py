"""Level-1 detector processing and optimal extraction."""

from collections.abc import Sequence

import ccdproc
from astropy import units as u
from astropy.nddata import CCDData, VarianceUncertainty

from .utils import ensure_variance, infer_half_width
from .cosmic import apply_cosmicray_correction
from .extraction import optimal_extract_traces


def apply_l1_corrections(
    ccd: CCDData,
    bias: CCDData | None = None,
    dark: CCDData | None = None,
    flat: CCDData | None = None,
    dark_scale: bool = False,
    exposure_key: str = "EXPTIME",
    exposure_unit: u.UnitBase = u.s,
    flat_min_value: float | None = None,
) -> CCDData:
    """Optionally bias-, dark-, and flat-correct an L0 image.

    Operations are applied in that order. Master calibration images are
    expected to be prepared upstream; in particular, a scaled dark should be
    bias-subtracted already.
    """
    result = ccd.copy()

    for name, calibration in (("bias", bias), ("dark", dark), ("flat", flat)):
        if calibration is not None and calibration.shape != result.shape:
            raise ValueError(
                f"master {name} shape {calibration.shape} does not match "
                f"science shape {result.shape}"
            )

    if bias is not None:
        result = ccdproc.subtract_bias(result, bias)

    if dark is not None:
        if dark_scale:
            result = ccdproc.subtract_dark(
                result,
                dark,
                exposure_time=exposure_key,
                exposure_unit=exposure_unit,
                scale=True,
            )
        else:
            result = ccdproc.subtract_dark(result, dark)

    if flat is not None:
        result = ccdproc.flat_correct(result, flat, min_value=flat_min_value)

    return result


def process_l1(
    ccd: CCDData,
    centers: Sequence[float],
    half_width: int | None = None,
    gain: float | None = None,
    read_noise: float | None = None,
    bias: CCDData | None = None,
    dark: CCDData | None = None,
    flat: CCDData | None = None,
    dark_scale: bool = False,
    exposure_key: str = "EXPTIME",
    flat_min_value: float | None = None,
):
    """Produce optimally extracted, pixel-coordinate spectra from an L0 image."""
    ccd = ensure_variance(ccd, gain, read_noise, bias=bias)
    ccd = apply_l1_corrections(
        ccd,
        bias=bias,
        dark=dark,
        flat=flat,
        dark_scale=dark_scale,
        exposure_key=exposure_key,
        flat_min_value=flat_min_value,
    )

    if ccd.uncertainty is None:
        raise ValueError("L1 processing produced an image with no uncertainty")

    variance = ccd.uncertainty.represent_as(VarianceUncertainty).array
    hw = half_width or infer_half_width(centers)
    return optimal_extract_traces(
        ccd.data,
        variance,
        centers,
        half_width=hw,
        mask=ccd.mask,
        unit=ccd.unit,
    )
