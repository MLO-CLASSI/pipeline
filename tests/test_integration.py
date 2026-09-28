import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData

from pipeline.l1 import process_l1
from pipeline.l1.io import write_l1_fits


def test_l1_synthetic_trace_end_to_end(tmp_path):
    ndisp = 40
    center = 10
    profile = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    amplitude = np.linspace(10.0, 20.0, ndisp)
    data = np.zeros((21, ndisp), dtype=float)
    data[center - 2:center + 3, :] = profile[:, None] * amplitude[None, :]
    ccd = CCDData(data, unit=u.adu)

    spectra = process_l1(
        ccd,
        centers=[center],
        half_width=4,
        gain=2.0,
        read_noise=3.0,
    )

    assert len(spectra) == 1
    np.testing.assert_allclose(
        spectra[0].flux.value,
        profile.sum() * amplitude,
        rtol=1e-10,
    )
    assert spectra[0].uncertainty is not None

    output = tmp_path / "l1.fits"
    write_l1_fits(
        output,
        spectra,
        [center],
        source_header=fits.Header({"OBJECT": "SYNTHETIC"}),
        source_path="synthetic_l0.fits",
        rebin=4,
    )

    with fits.open(output) as hdul:
        assert hdul[0].header["PROCLVL"] == 1
        assert hdul[0].header["REBIN"] == 4
        trace = hdul["TRACE1"]
        assert len(trace.data) == ndisp // 4
        np.testing.assert_allclose(trace.data["PIXEL"], np.arange(1.5, ndisp, 4.0))
        assert np.all(np.isfinite(trace.data["COUNTS"]))
        assert np.all(np.isfinite(trace.data["SIGMA"]))
