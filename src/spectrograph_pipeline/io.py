"""FITS input/output helpers."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.table import Table


def _resolve_hdu(hdul: fits.HDUList, extension: str | int):
    try:
        return hdul[extension]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"FITS extension {extension!r} was not found") from exc


def read_image(
    path: str | Path,
    data_ext: str | int = 0,
    variance_ext: str | int | None = None,
    mask_ext: str | int | None = None,
):
    """Read science data and optional variance/mask planes from a FITS file."""
    with fits.open(path, memmap=False) as hdul:
        data_hdu = _resolve_hdu(hdul, data_ext)
        if data_hdu.data is None:
            raise ValueError(f"FITS extension {data_ext!r} contains no image data")

        data = np.asarray(data_hdu.data, dtype=float)
        header = data_hdu.header.copy()

        variance = None
        if variance_ext is not None:
            variance_hdu = _resolve_hdu(hdul, variance_ext)
            variance = np.asarray(variance_hdu.data, dtype=float)

        mask = None
        if mask_ext is not None:
            mask_hdu = _resolve_hdu(hdul, mask_ext)
            mask = np.asarray(mask_hdu.data, dtype=bool)

    return data, variance, mask, header


def write_extracted_fits(
    path: str | Path,
    spectra,
    centers,
    source_header: fits.Header,
    source_path: str | Path,
    overwrite: bool = False,
) -> None:
    """Write one binary-table extension per extracted fiber."""
    primary_header = source_header.copy()
    primary_header["NFIBERS"] = (len(spectra), "Number of extracted fibers")
    primary_header["EXTRACT"] = ("HORNE", "Extraction algorithm")
    primary_header.add_history(f"Optimal extraction source: {source_path}")

    hdus = [fits.PrimaryHDU(header=primary_header)]

    for fiber_id, (spectrum, center) in enumerate(zip(spectra, centers, strict=True), start=1):
        flux = np.asarray(spectrum.flux.value, dtype=float)
        sigma = np.asarray(
            spectrum.uncertainty.represent_as(StdDevUncertainty).array,
            dtype=float,
        )
        pixel = np.arange(flux.size, dtype=np.int32)
        mask = spectrum.mask
        if mask is None:
            mask = np.zeros(flux.size, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
        mask |= ~np.isfinite(flux) | ~np.isfinite(sigma)

        table = Table()
        table["PIXEL"] = pixel
        table["FLUX"] = flux
        table["SIGMA"] = sigma
        table["MASK"] = mask
        table["FLUX"].unit = spectrum.flux.unit
        table["SIGMA"].unit = spectrum.flux.unit

        hdu = fits.BinTableHDU(table, name=f"FIBER{fiber_id}")
        hdu.header["FIBERID"] = fiber_id
        hdu.header["TRACECEN"] = (float(center), "Flat trace center [pixel]")
        hdu.header["EXTRACT"] = "HORNE"
        hdus.append(hdu)

    fits.HDUList(hdus).writeto(path, overwrite=overwrite, checksum=True)
