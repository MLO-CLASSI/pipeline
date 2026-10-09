import numpy as np
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from astropy.table import Table
from specutils import Spectrum

from pipeline.l3 import Bandpass, mangle
from specmangle import synthetic_ab_magnitude


def test_l3_mangling_delegates_to_specmangle():
    wavelength = np.linspace(4000.0, 7000.0, 301) * u.AA
    flux = np.full(wavelength.size, 1.0e-15) * u.erg / u.s / u.cm**2 / u.AA
    sigma = np.full(wavelength.size, 1.0e-17) * flux.unit
    spectrum = Spectrum(
        spectral_axis=wavelength,
        flux=flux,
        uncertainty=StdDevUncertainty(sigma),
    )

    bandpasses = {
        "B": Bandpass(
            "test/B",
            np.array([4000.0, 4200.0, 4600.0, 4800.0]) * u.AA,
            np.array([0.0, 1.0, 1.0, 0.0]),
            "photon",
        ),
        "R": Bandpass(
            "test/R",
            np.array([5900.0, 6100.0, 6600.0, 6800.0]) * u.AA,
            np.array([0.0, 1.0, 1.0, 0.0]),
            "photon",
        ),
    }

    target_mag = np.array(
        [
            synthetic_ab_magnitude(spectrum, bandpasses[band]).magnitude
            for band in ("B", "R")
        ]
    ) - 0.25

    result = mangle(
        spectrum,
        Table(
            {
                "band": ["B", "R"],
                "mag": target_mag,
                "mag_err": [0.02, 0.02],
            }
        ),
        bandpasses=bandpasses,
    )

    np.testing.assert_allclose(result.photometry["synthetic_mag"], target_mag, atol=1e-8)
    assert np.all(result.correction > 1.0)
    np.testing.assert_allclose(
        result.spectrum.uncertainty.quantity.to_value(flux.unit),
        (sigma * result.correction).to_value(flux.unit),
    )
    assert result.parameter_covariance is not None
