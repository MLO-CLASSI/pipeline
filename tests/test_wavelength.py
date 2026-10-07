import asdf
import numpy as np
import pytest
from astropy import units as u

from pipeline.l2 import (
    fit_arc_wavelength_solution,
    fit_sky_refinement,
    read_wavesol,
    write_wavesol,
)


def _arc_solution(trace_id=1, offset=0.0):
    pixels = np.array([100.0, 450.0, 900.0, 1350.0, 1800.0])
    ref_pixel = 1000.0
    x = pixels - ref_pixel
    wavelengths = (6000.0 + offset) - 1.15 * x + 2.5e-5 * x**2
    return fit_arc_wavelength_solution(
        trace_id,
        pixels,
        wavelengths * u.AA,
        (0, 2048),
        degree=2,
        ref_pixel=ref_pixel,
        line_ids=["L1", "L2", "L3", "L4", "L5"],
    )


def test_arc_fit_recovers_polynomial_and_reference_pixel():
    result = _arc_solution()
    test_pixels = np.array([0.0, 512.0, 1000.0, 1536.0, 2047.0])
    x = test_pixels - 1000.0
    expected = 6000.0 - 1.15 * x + 2.5e-5 * x**2

    assert result.reference_pixel == pytest.approx(1000.0)
    np.testing.assert_allclose(result.solution.pix_to_wav(test_pixels), expected, atol=1e-9)
    assert result.rms_wavelength.to_value(u.AA) < 1e-9
    assert result.rms_pixel < 1e-9


def test_arc_fit_requires_enough_lines():
    with pytest.raises(ValueError, match=r"degree \+ 1"):
        fit_arc_wavelength_solution(
            1,
            [100.0, 200.0],
            [5000.0, 5100.0] * u.AA,
            (0, 2048),
            degree=2,
        )


def test_arc_fit_can_store_rejected_lines_without_using_them():
    pixels = np.array([100.0, 500.0, 1000.0, 1500.0])
    wavelengths = 6000.0 - 1.2 * (pixels - 1000.0)
    wavelengths[-1] += 50.0
    result = fit_arc_wavelength_solution(
        1,
        pixels,
        wavelengths * u.AA,
        (0, 2048),
        degree=1,
        ref_pixel=1000.0,
        used=[True, True, True, False],
    )

    np.testing.assert_array_equal(result.used, [True, True, True, False])
    assert abs(result.wavelength_residuals[-1].to_value(u.AA)) > 40.0
    assert result.rms_wavelength.to_value(u.AA) < 1e-9


def test_wavesol_asdf_round_trip(tmp_path):
    path = tmp_path / "classi_wavesol.asdf"
    original = [_arc_solution(1), _arc_solution(2, offset=0.7)]

    write_wavesol(
        path,
        original,
        arc_file="arc_20261002.fits",
        lamps=["HgI", "NeI"],
    )

    with asdf.open(path, lazy_load=False) as af:
        root = af["classi_wavesol"]
        assert root["format"] == "CLASSI-WAVESOL"
        assert root["version"] == 1
        assert root["wavelength_medium"] == "vacuum"
        assert root["pixel_origin"] == 0
        assert root["dispersion_axis"] == 1
        assert root["n_traces"] == 2
        assert root["arc_file"] == "arc_20261002.fits"
        assert list(root["lamps"]) == ["HgI", "NeI"]

        trace = root["traces"][0]
        assert trace["trace_id"] == 1
        assert trace["wavelength_solution"]["gwcs"].bounding_box is not None
        assert trace["wavelength_solution"]["wave_air"] is False
        assert set(trace["lines"]) == {
            "pixel",
            "wavelength",
            "wavelength_residual",
            "pixel_residual",
            "used",
            "line_id",
        }

    restored = read_wavesol(path)
    assert set(restored) == {1, 2}
    sample = np.linspace(0.0, 2047.0, 25)
    for trace_id, expected in ((1, original[0]), (2, original[1])):
        result = restored[trace_id]
        np.testing.assert_allclose(
            result.solution.pix_to_wav(sample),
            expected.solution.pix_to_wav(sample),
            rtol=0.0,
            atol=1e-10,
        )
        assert result.solution.bounds_pix == expected.solution.bounds_pix
        assert result.line_ids == expected.line_ids
        np.testing.assert_array_equal(result.used, expected.used)


def test_wavesol_rejects_mixed_wavelength_medium(tmp_path):
    vacuum = _arc_solution(1)
    air = fit_arc_wavelength_solution(
        2,
        vacuum.line_pixels,
        vacuum.line_wavelengths,
        vacuum.solution.bounds_pix,
        degree=2,
        ref_pixel=vacuum.reference_pixel,
        wave_air=True,
    )

    with pytest.raises(ValueError, match="same wavelength medium"):
        write_wavesol(tmp_path / "bad.asdf", [vacuum, air])


def test_wavesol_does_not_overwrite_by_default(tmp_path):
    path = tmp_path / "classi_wavesol.asdf"
    write_wavesol(path, [_arc_solution()])

    with pytest.raises(FileExistsError):
        write_wavesol(path, [_arc_solution()])

    write_wavesol(path, [_arc_solution()], overwrite=True)
    assert path.is_file()


def test_sky_shift_refinement_recovers_pixel_offset():
    baseline = _arc_solution().solution
    master_pixels = np.array([400.0, 900.0, 1500.0])
    wavelengths = baseline.pix_to_wav(master_pixels) * baseline.unit
    true_shift = 0.63
    observed_pixels = master_pixels - true_shift

    refinement = fit_sky_refinement(
        baseline,
        observed_pixels,
        wavelengths,
        mode="shift",
    )

    assert refinement.mode == "shift"
    assert refinement.shift == pytest.approx(true_shift, abs=1e-10)
    assert refinement.stretch == 1.0
    assert refinement.rms_pixel < 1e-10
    np.testing.assert_allclose(refinement.master_pixel(observed_pixels), master_pixels, atol=1e-10)


@pytest.mark.parametrize("master_pixel", [-1.0, 2048.0])
def test_sky_refinement_rejects_wavelengths_outside_detector_edges(master_pixel):
    baseline = _arc_solution().solution
    wavelength = baseline.pix_to_wav([master_pixel]) * baseline.unit

    with pytest.raises(ValueError, match="outside the baseline solution"):
        fit_sky_refinement(baseline, [1000.0], wavelength, mode="shift")


def test_sky_affine_refinement_recovers_shift_and_stretch():
    baseline = _arc_solution().solution
    reference = 1000.0
    master_pixels = np.array([250.0, 650.0, 1200.0, 1750.0])
    wavelengths = baseline.pix_to_wav(master_pixels) * baseline.unit
    true_shift = -0.35
    true_stretch = 1.0009
    observed_pixels = reference + (master_pixels - reference - true_shift) / true_stretch

    refinement = fit_sky_refinement(
        baseline,
        observed_pixels,
        wavelengths,
        mode="affine",
    )

    assert refinement.mode == "affine"
    assert refinement.shift == pytest.approx(true_shift, abs=1e-10)
    assert refinement.stretch == pytest.approx(true_stretch, abs=1e-12)
    assert refinement.rms_pixel < 1e-10

    refined = refinement.refined_solution(baseline)
    np.testing.assert_allclose(
        refined.pix_to_wav(observed_pixels),
        wavelengths.to_value(baseline.unit),
        atol=1e-9,
    )


def test_sky_auto_uses_shift_for_one_line_and_affine_for_multiple():
    baseline = _arc_solution().solution

    master = np.array([700.0])
    wavelengths = baseline.pix_to_wav(master) * baseline.unit
    one = fit_sky_refinement(baseline, master - 0.4, wavelengths, mode="auto")
    assert one.mode == "shift"

    master = np.array([500.0, 1500.0])
    wavelengths = baseline.pix_to_wav(master) * baseline.unit
    two = fit_sky_refinement(baseline, master - 0.4, wavelengths, mode="auto")
    assert two.mode == "affine"


def test_sky_affine_requires_two_lines():
    baseline = _arc_solution().solution
    wavelength = baseline.pix_to_wav([700.0]) * baseline.unit

    with pytest.raises(ValueError, match="at least two"):
        fit_sky_refinement(baseline, [699.5], wavelength, mode="affine")


def test_sky_refinement_uses_centroid_uncertainties_for_covariance():
    baseline = _arc_solution().solution
    master_pixels = np.array([300.0, 800.0, 1300.0, 1800.0])
    wavelengths = baseline.pix_to_wav(master_pixels) * baseline.unit
    observed = master_pixels - 0.5

    result = fit_sky_refinement(
        baseline,
        observed,
        wavelengths,
        pixel_uncertainty=np.full(4, 0.08),
        mode="affine",
    )

    assert result.covariance is not None
    assert result.covariance.shape == (2, 2)
    assert np.all(np.isfinite(result.covariance))
