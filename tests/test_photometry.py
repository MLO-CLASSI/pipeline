import numpy as np
import pytest
from astropy import units as u
from synphot import SourceSpectrum, SpectralElement, units as syn_units
from synphot.models import Empirical1D

from pipeline.l3 import (
    PhotometricPoint,
    apply_photometric_correction,
    fit_photometric_correction,
    synthetic_magnitude,
)


def _bandpass(center, width):
    half_width = width / 2
    wavelength = np.array(
        [
            center - half_width - 50,
            center - half_width,
            center + half_width,
            center + half_width + 50,
        ]
    ) * u.AA
    throughput = np.array([0.0, 1.0, 1.0, 0.0])
    return SpectralElement(
        Empirical1D,
        points=wavelength,
        lookup_table=throughput,
    )


def _spectrum():
    wavelength = np.linspace(3500.0, 7500.0, 1001) * u.AA
    flux = 1.0e-15 * (wavelength.to_value(u.AA) / 5500.0) ** -1.2 * syn_units.FLAM
    return wavelength, flux


def test_three_band_fit_recovers_known_quadratic_correction():
    wavelength, flux = _spectrum()
    bandpasses = {
        "B": _bandpass(4400.0, 700.0),
        "V": _bandpass(5500.0, 800.0),
        "R": _bandpass(6500.0, 900.0),
    }

    reference = 5400.0
    x = np.log(wavelength.to_value(u.AA) / reference)
    expected = np.exp(np.polynomial.polynomial.polyval(x, [np.log(1.25), 0.18, -0.12]))
    target_flux = flux * expected

    photometry = [
        PhotometricPoint(
            band,
            synthetic_magnitude(wavelength, target_flux, bandpasses[band], system="abmag"),
            0.01,
            system="abmag",
        )
        for band in ("B", "V", "R")
    ]

    fitted = fit_photometric_correction(
        wavelength,
        flux,
        photometry,
        bandpasses=bandpasses,
    )

    assert fitted.degree == 2
    np.testing.assert_allclose(fitted.synthetic_after, [p.magnitude for p in photometry], atol=1e-7)
    np.testing.assert_allclose(fitted.evaluate(wavelength), expected, rtol=2e-5, atol=0.0)
    assert fitted.covariance.shape == (3, 3)


def test_one_band_fit_is_gray_scaling():
    wavelength, flux = _spectrum()
    bandpass = _bandpass(5500.0, 800.0)
    target_flux = 1.7 * flux
    magnitude = synthetic_magnitude(wavelength, target_flux, bandpass, system="abmag")

    fitted = fit_photometric_correction(
        wavelength,
        flux,
        [PhotometricPoint("V", magnitude, 0.02, system="abmag")],
        bandpasses={"V": bandpass},
    )

    assert fitted.degree == 0
    np.testing.assert_allclose(fitted.evaluate(wavelength), 1.7, rtol=1e-7)


def test_two_band_fit_uses_linear_log_wavelength_correction():
    wavelength, flux = _spectrum()
    bandpasses = {
        "B": _bandpass(4400.0, 700.0),
        "R": _bandpass(6500.0, 900.0),
    }
    x = np.log(wavelength.to_value(u.AA) / 5400.0)
    expected = np.exp(np.polynomial.polynomial.polyval(x, [np.log(1.1), 0.2]))
    target_flux = flux * expected

    photometry = [
        PhotometricPoint(
            band,
            synthetic_magnitude(wavelength, target_flux, bandpasses[band], system="abmag"),
            0.01,
            system="abmag",
        )
        for band in ("B", "R")
    ]
    fitted = fit_photometric_correction(
        wavelength,
        flux,
        photometry,
        bandpasses=bandpasses,
    )

    assert fitted.degree == 1
    np.testing.assert_allclose(fitted.evaluate(wavelength), expected, rtol=2e-5)


def test_vega_magnitude_accepts_explicit_vega_reference():
    wavelength, flux = _spectrum()
    bandpass = _bandpass(5500.0, 800.0)
    vega = SourceSpectrum(
        Empirical1D,
        points=wavelength,
        lookup_table=np.full(wavelength.size, 3.5e-9) * syn_units.FLAM,
        keep_neg=True,
    )

    magnitude = synthetic_magnitude(
        wavelength,
        flux,
        bandpass,
        system="vegamag",
        vega_spectrum=vega,
    )

    assert np.isfinite(magnitude)


def test_partial_bandpass_coverage_is_rejected():
    wavelength = np.linspace(5000.0, 6000.0, 101) * u.AA
    flux = np.ones(wavelength.size) * 1.0e-15 * syn_units.FLAM
    bandpass = _bandpass(5500.0, 1600.0)

    with pytest.raises(ValueError, match="does not fully cover"):
        synthetic_magnitude(wavelength, flux, bandpass, system="abmag")


def test_degree_must_be_supported_by_number_of_points():
    wavelength, flux = _spectrum()
    bandpass = _bandpass(5500.0, 800.0)
    point = PhotometricPoint("V", 18.0, 0.03, system="abmag")

    with pytest.raises(ValueError, match="degree must be smaller"):
        fit_photometric_correction(
            wavelength,
            flux,
            [point],
            degree=1,
            bandpasses={"V": bandpass},
        )


def test_apply_correction_scales_flux_and_statistical_uncertainty():
    wavelength, flux = _spectrum()
    sigma = np.full(wavelength.size, 2.0e-17) * syn_units.FLAM
    bandpass = _bandpass(5500.0, 800.0)
    magnitude = synthetic_magnitude(wavelength, 1.4 * flux, bandpass, system="abmag")
    fitted = fit_photometric_correction(
        wavelength,
        flux,
        [PhotometricPoint("V", magnitude, 0.02, system="abmag")],
        bandpasses={"V": bandpass},
    )

    corrected_flux, corrected_sigma = apply_photometric_correction(
        wavelength,
        flux,
        sigma,
        fitted,
    )

    np.testing.assert_allclose(corrected_flux.value, 1.4 * flux.value, rtol=1e-7)
    np.testing.assert_allclose(corrected_sigma.value, 1.4 * sigma.value, rtol=1e-7)
