"""Broadband photometric anchoring for calibrated spectra.

This module fits a smooth multiplicative correction to an already wavelength-
and flux-calibrated spectrum. The correction is constrained by synthetic
photometry through the full supplied bandpasses rather than by treating each
photometric point as monochromatic.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from astropy import units as u
from scipy.optimize import least_squares
from synphot import Observation, SourceSpectrum, SpectralElement, units as syn_units
from synphot.models import Empirical1D


_FILTER_ALIASES = {
    "B": "johnson_b",
    "V": "johnson_v",
    "R": "cousins_r",
    "RC": "cousins_r",
    "I": "cousins_i",
    "IC": "cousins_i",
}


@dataclass(frozen=True)
class PhotometricPoint:
    """One broadband magnitude used to anchor a spectrum."""

    band: str
    magnitude: float
    uncertainty: float
    system: str = "vegamag"


@dataclass(frozen=True)
class PhotometricCorrection:
    """Fitted log-polynomial multiplicative correction."""

    coefficients: np.ndarray
    covariance: np.ndarray
    reference_wavelength: u.Quantity
    degree: int
    bands: tuple[str, ...]
    synthetic_before: np.ndarray
    synthetic_after: np.ndarray

    def evaluate(self, wavelength: u.Quantity) -> np.ndarray:
        """Evaluate the positive multiplicative correction at wavelength."""
        wavelength = u.Quantity(wavelength).to(u.AA)
        x = np.log(wavelength.to_value(u.AA) / self.reference_wavelength.to_value(u.AA))
        log_correction = np.polynomial.polynomial.polyval(x, self.coefficients)
        return np.exp(log_correction)


@lru_cache(maxsize=1)
def _default_vega_spectrum() -> SourceSpectrum:
    return SourceSpectrum.from_vega()


def _bandpass_for(
    band: str,
    bandpasses: Mapping[str, SpectralElement] | None,
) -> SpectralElement:
    if bandpasses is not None:
        try:
            return bandpasses[band]
        except KeyError as exc:
            raise ValueError(f"no bandpass supplied for {band!r}") from exc

    filter_name = _FILTER_ALIASES.get(band.upper(), band)
    return SpectralElement.from_filter(filter_name)


def _source_spectrum(wavelength: u.Quantity, flux: u.Quantity) -> SourceSpectrum:
    wavelength = u.Quantity(wavelength).to(u.AA)
    flux = u.Quantity(flux)

    if wavelength.ndim != 1 or flux.ndim != 1 or wavelength.shape != flux.shape:
        raise ValueError("wavelength and flux must be matching one-dimensional arrays")
    if wavelength.size < 2:
        raise ValueError("spectrum must contain at least two samples")
    if not np.all(np.isfinite(wavelength.value)) or not np.all(np.isfinite(flux.value)):
        raise ValueError("wavelength and flux must be finite")
    if np.any(np.diff(wavelength.value) <= 0):
        raise ValueError("wavelength must be strictly increasing")

    return SourceSpectrum(
        Empirical1D,
        points=wavelength,
        lookup_table=flux,
        keep_neg=True,
    )


def synthetic_magnitude(
    wavelength: u.Quantity,
    flux: u.Quantity,
    bandpass: SpectralElement,
    system: str = "vegamag",
    vega_spectrum: SourceSpectrum | None = None,
) -> float:
    """Calculate a synthetic broadband magnitude for a calibrated spectrum.

    The spectrum must fully cover the nonzero portion of the bandpass. Partial
    band coverage is rejected rather than extrapolated.
    """
    source = _source_spectrum(wavelength, flux)
    if bandpass.check_overlap(source) != "full":
        raise ValueError("spectrum does not fully cover the photometric bandpass")

    observation = Observation(source, bandpass)
    system = system.lower()

    if system in {"ab", "abmag"}:
        return float(observation.effstim(flux_unit=u.ABmag).value)
    if system in {"vega", "vegamag"}:
        if vega_spectrum is None:
            vega_spectrum = _default_vega_spectrum()
        return float(
            observation.effstim(
                flux_unit=syn_units.VEGAMAG,
                vegaspec=vega_spectrum,
            ).value
        )

    raise ValueError(f"unsupported magnitude system {system!r}")


def fit_photometric_correction(
    wavelength: u.Quantity,
    flux: u.Quantity,
    photometry: Sequence[PhotometricPoint],
    degree: int | None = None,
    bandpasses: Mapping[str, SpectralElement] | None = None,
    vega_spectrum: SourceSpectrum | None = None,
) -> PhotometricCorrection:
    """Fit a log-polynomial correction to broadband photometry.

    The fitted model is::

        C(lambda) = exp(sum(a_i * log(lambda / lambda_ref)**i))

    With ``degree=None``, the degree is selected from the available information:
    zero for one band, one for two bands, and two for three or more bands.
    """
    photometry = tuple(photometry)
    if not photometry:
        raise ValueError("at least one photometric point is required")

    for point in photometry:
        if not np.isfinite(point.magnitude):
            raise ValueError(f"non-finite magnitude for band {point.band!r}")
        if not np.isfinite(point.uncertainty) or point.uncertainty <= 0:
            raise ValueError(f"photometric uncertainty for band {point.band!r} must be positive")

    if degree is None:
        degree = min(2, len(photometry) - 1)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if degree >= len(photometry):
        raise ValueError("polynomial degree must be smaller than the number of photometric points")

    wavelength = u.Quantity(wavelength).to(u.AA)
    flux = u.Quantity(flux)
    _source_spectrum(wavelength, flux)

    resolved_bandpasses = [
        _bandpass_for(point.band, bandpasses)
        for point in photometry
    ]
    pivot_wavelengths = np.array(
        [bandpass.pivot().to_value(u.AA) for bandpass in resolved_bandpasses],
        dtype=float,
    )
    reference_wavelength = np.exp(np.mean(np.log(pivot_wavelengths))) * u.AA

    observed = np.array([point.magnitude for point in photometry], dtype=float)
    errors = np.array([point.uncertainty for point in photometry], dtype=float)

    def calculate_magnitudes(test_flux: u.Quantity) -> np.ndarray:
        return np.array(
            [
                synthetic_magnitude(
                    wavelength,
                    test_flux,
                    bandpass,
                    system=point.system,
                    vega_spectrum=vega_spectrum,
                )
                for point, bandpass in zip(
                    photometry,
                    resolved_bandpasses,
                    strict=True,
                )
            ],
            dtype=float,
        )

    synthetic_before = calculate_magnitudes(flux)
    x = np.log(wavelength.to_value(u.AA) / reference_wavelength.to_value(u.AA))

    initial = np.zeros(degree + 1, dtype=float)
    weights = 1.0 / errors**2
    initial[0] = 0.4 * np.log(10.0) * np.average(
        synthetic_before - observed,
        weights=weights,
    )

    def residuals(coefficients: np.ndarray) -> np.ndarray:
        correction = np.exp(np.polynomial.polynomial.polyval(x, coefficients))
        synthetic = calculate_magnitudes(flux * correction)
        return (synthetic - observed) / errors

    fit = least_squares(residuals, initial)
    if not fit.success:
        raise RuntimeError(f"photometric correction fit failed: {fit.message}")

    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
    correction = np.exp(np.polynomial.polynomial.polyval(x, fit.x))
    synthetic_after = calculate_magnitudes(flux * correction)

    return PhotometricCorrection(
        coefficients=np.asarray(fit.x, dtype=float),
        covariance=np.asarray(covariance, dtype=float),
        reference_wavelength=reference_wavelength,
        degree=degree,
        bands=tuple(point.band for point in photometry),
        synthetic_before=synthetic_before,
        synthetic_after=synthetic_after,
    )


def apply_photometric_correction(
    wavelength: u.Quantity,
    flux: u.Quantity,
    uncertainty: u.Quantity | None,
    correction: PhotometricCorrection,
) -> tuple[u.Quantity, u.Quantity | None]:
    """Apply a fitted correction to flux and its statistical uncertainty.

    This scales the existing per-sample uncertainty only. The correlated
    uncertainty in the fitted photometric correction is represented by
    ``PhotometricCorrection.covariance`` and is intentionally not folded into
    the per-sample uncertainty array.
    """
    factor = correction.evaluate(wavelength)
    corrected_flux = u.Quantity(flux) * factor
    corrected_uncertainty = None
    if uncertainty is not None:
        corrected_uncertainty = u.Quantity(uncertainty) * factor
    return corrected_flux, corrected_uncertainty
