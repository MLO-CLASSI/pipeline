"""FITS output helpers for pipeline data products."""

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.table import Table


def write_l1_fits(
    path: str | Path,
    spectra,
    centers,
    source_header: fits.Header,
    source_path: str | Path,
    bias_path: str | Path | None = None,
    dark_path: str | Path | None = None,
    flat_path: str | Path | None = None,
    dark_scaled: bool = False,
    rebin: int = 1,
    overwrite: bool = False,
) -> None:
    """Write an L1 product with one pixel-coordinate table per trace."""
    if rebin < 1:
        raise ValueError("rebin must be a positive integer")

    primary_header = source_header.copy()
    primary_header["PROCLVL"] = (1, "Pipeline processing level")
    primary_header["PROCTYPE"] = ("L1", "Pipeline product type")
    primary_header["NTRACE"] = (len(spectra), "Number of extracted traces")
    primary_header["EXTRACT"] = ("HORNE", "Extraction algorithm")
    primary_header["PIXORIG"] = (0, "Origin of PIXEL coordinates")
    primary_header["REBIN"] = (rebin, "Native dispersion pixels per output bin")
    primary_header["BIASCOR"] = (bias_path is not None, "Bias correction applied")
    primary_header["DARKCOR"] = (dark_path is not None, "Dark correction applied")
    primary_header["FLATCOR"] = (flat_path is not None, "Flat correction applied")
    primary_header["DARKSCL"] = (bool(dark_scaled), "Dark scaled by exposure time")
    primary_header["WAVECAL"] = (False, "Wavelength calibration applied")
    primary_header["FLUXCAL"] = (False, "Spectrophotometric calibration applied")
    primary_header.add_history(f"L1 source: {source_path}")

    if bias_path is not None:
        primary_header.add_history(f"Master bias: {bias_path}")
    if dark_path is not None:
        primary_header.add_history(f"Master dark: {dark_path}")
    if flat_path is not None:
        primary_header.add_history(f"Master flat: {flat_path}")

    hdus = [fits.PrimaryHDU(header=primary_header)]

    for trace_id, (spectrum, center) in enumerate(
        zip(spectra, centers, strict=True),
        start=1,
    ):
        counts = np.asarray(spectrum.flux.value, dtype=float)
        sigma = np.asarray(spectrum.uncertainty.represent_as(StdDevUncertainty).array, dtype=float)
        pixel = np.arange(counts.size, dtype=float)
        mask = spectrum.mask
        if mask is None:
            mask = np.zeros(counts.size, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
        mask |= ~np.isfinite(counts) | ~np.isfinite(sigma)

        n_trimmed = counts.size % rebin
        n_used = counts.size - n_trimmed
        if n_used == 0:
            raise ValueError(
                f"rebin factor {rebin} exceeds spectrum length {counts.size}"
            )
        if rebin > 1:
            shape = (-1, rebin)
            pixel = pixel[:n_used].reshape(shape).mean(axis=1)
            valid = ~mask[:n_used]
            counts = np.where(valid, counts[:n_used], 0.0).reshape(shape).sum(axis=1)
            sigma = np.sqrt(
                np.where(valid, sigma[:n_used] ** 2, 0.0).reshape(shape).sum(axis=1)
            )
            mask = mask[:n_used].reshape(shape).any(axis=1)

        table = Table()
        table["PIXEL"] = pixel
        table["COUNTS"] = counts
        table["SIGMA"] = sigma
        table["MASK"] = mask
        table["COUNTS"].unit = spectrum.flux.unit
        table["SIGMA"].unit = spectrum.flux.unit

        hdu = fits.BinTableHDU(table, name=f"TRACE{trace_id}")
        hdu.header["PROCLVL"] = 1
        hdu.header["TRACEID"] = trace_id
        hdu.header["PIXORIG"] = (0, "Origin of PIXEL coordinates")
        hdu.header["REBIN"] = (rebin, "Native dispersion pixels per output bin")
        hdu.header["NTRIM"] = (n_trimmed, "Trailing native pixels omitted by rebinning")
        hdu.header["TRACECEN"] = (float(center), "Flat trace center [pixel]")
        hdu.header["EXTRACT"] = "BOXCAR"
        hdu.header["WAVECAL"] = False
        hdu.header["FLUXCAL"] = False
        hdus.append(hdu)

    fits.HDUList(hdus).writeto(path, overwrite=overwrite, checksum=True)
