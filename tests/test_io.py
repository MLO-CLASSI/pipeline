from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum

from pipeline.l1.io import write_l1_fits


def make_spectrum(counts, sigma, mask=None):
    counts = np.asarray(counts, dtype=float)
    return Spectrum(
        flux=counts * u.adu,
        spectral_axis=np.arange(counts.size) * u.pix,
        uncertainty=StdDevUncertainty(np.asarray(sigma, dtype=float)),
        mask=None if mask is None else np.asarray(mask, dtype=bool),
    )


def test_write_l1_fits_preserves_unbinned_samples_and_metadata(tmp_path):
    output = tmp_path / "l1.fits"
    spectrum = make_spectrum([10, 20, 30], [1, 2, 3])
    source_header = fits.Header({"OBJECT": "TEST"})

    write_l1_fits(
        output,
        [spectrum],
        [12.5],
        source_header=source_header,
        source_path="science_l0.fits",
    )

    with fits.open(output) as hdul:
        assert len(hdul) == 2
        assert hdul[0].header["OBJECT"] == "TEST"
        assert hdul[0].header["PROCLVL"] == 1
        assert hdul[0].header["PROCTYPE"] == "L1"
        assert hdul[0].header["NTRACE"] == 1
        assert hdul[0].header["PIXORIG"] == 0
        assert hdul[0].header["REBIN"] == 1
        assert not hdul[0].header["WAVECAL"]
        assert not hdul[0].header["FLUXCAL"]

        trace = hdul["TRACE1"]
        np.testing.assert_allclose(trace.data["PIXEL"], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(trace.data["COUNTS"], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(trace.data["SIGMA"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(trace.data["MASK"], [False, False, False])
        assert trace.header["TRACECEN"] == pytest.approx(12.5)
        assert trace.header["EXTRACT"] == "BOXCAR"
        assert trace.header["REBIN"] == 1
        assert trace.header["NTRIM"] == 0
        assert trace.columns["COUNTS"].unit == "adu"
        assert trace.columns["SIGMA"].unit == "adu"


def test_write_l1_fits_rebins_counts_uncertainty_mask_and_pixel_coordinate(tmp_path):
    output = tmp_path / "rebinned.fits"
    spectrum = make_spectrum(
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        mask=[False, True, False, False, False],
    )

    write_l1_fits(
        output,
        [spectrum],
        [7.0],
        source_header=fits.Header(),
        source_path=Path("input.fits"),
        rebin=2,
    )

    with fits.open(output) as hdul:
        trace = hdul["TRACE1"]
        np.testing.assert_allclose(trace.data["PIXEL"], [0.5, 2.5])
        np.testing.assert_allclose(trace.data["COUNTS"], [1.0, 7.0])
        np.testing.assert_allclose(trace.data["SIGMA"], [1.0, 5.0])
        np.testing.assert_array_equal(trace.data["MASK"], [True, False])
        assert hdul[0].header["REBIN"] == 2
        assert trace.header["REBIN"] == 2
        assert trace.header["NTRIM"] == 1


def test_write_l1_fits_masks_nonfinite_samples(tmp_path):
    output = tmp_path / "nonfinite.fits"
    spectrum = make_spectrum([1.0, np.nan, 3.0], [1.0, 2.0, np.inf])

    write_l1_fits(
        output,
        [spectrum],
        [4.0],
        source_header=fits.Header(),
        source_path="input.fits",
    )

    with fits.open(output) as hdul:
        np.testing.assert_array_equal(hdul["TRACE1"].data["MASK"], [False, True, True])


@pytest.mark.parametrize("rebin", [0, -1])
def test_write_l1_fits_rejects_nonpositive_rebin(tmp_path, rebin):
    spectrum = make_spectrum([1, 2], [1, 1])

    with pytest.raises(ValueError, match="rebin must be a positive integer"):
        write_l1_fits(
            tmp_path / "bad.fits",
            [spectrum],
            [1.0],
            source_header=fits.Header(),
            source_path="input.fits",
            rebin=rebin,
        )


def test_write_l1_fits_rejects_rebin_larger_than_spectrum(tmp_path):
    spectrum = make_spectrum([1, 2], [1, 1])

    with pytest.raises(ValueError, match="exceeds spectrum length"):
        write_l1_fits(
            tmp_path / "bad.fits",
            [spectrum],
            [1.0],
            source_header=fits.Header(),
            source_path="input.fits",
            rebin=3,
        )


def test_write_l1_fits_rejects_mismatched_spectra_and_centers(tmp_path):
    spectrum = make_spectrum([1, 2], [1, 1])

    with pytest.raises(ValueError):
        write_l1_fits(
            tmp_path / "bad.fits",
            [spectrum],
            [],
            source_header=fits.Header(),
            source_path="input.fits",
        )


def test_write_l1_fits_records_calibration_provenance(tmp_path):
    output = tmp_path / "provenance.fits"
    spectrum = make_spectrum([1, 2], [1, 1])

    write_l1_fits(
        output,
        [spectrum],
        [1.0],
        source_header=fits.Header(),
        source_path="science.fits",
        bias_path="bias.fits",
        dark_path="dark.fits",
        flat_path="flat.fits",
        dark_scaled=True,
    )

    with fits.open(output) as hdul:
        header = hdul[0].header
        assert header["BIASCOR"]
        assert header["DARKCOR"]
        assert header["FLATCOR"]
        assert header["DARKSCL"]
        history = header["HISTORY"]
        assert "L1 source: science.fits" in history
        assert "Master bias: bias.fits" in history
        assert "Master dark: dark.fits" in history
        assert "Master flat: flat.fits" in history


@pytest.mark.xfail(
    reason="Current HEAD writes EXTRACT=HORNE in the primary header while L1 uses BoxcarExtract",
    strict=True,
)
def test_primary_header_reports_boxcar_extraction(tmp_path):
    output = tmp_path / "extract.fits"
    spectrum = make_spectrum([1, 2], [1, 1])

    write_l1_fits(
        output,
        [spectrum],
        [1.0],
        source_header=fits.Header(),
        source_path="input.fits",
    )

    with fits.open(output) as hdul:
        assert hdul[0].header["EXTRACT"] == "BOXCAR"
