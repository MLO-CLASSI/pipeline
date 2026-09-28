import argparse

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.nddata import VarianceUncertainty

import pipeline.cli as cli


def test_extension_parses_integer_or_name():
    assert cli._extension("0") == 0
    assert cli._extension("SCI") == "SCI"


def test_get_centers_parses_explicit_centers():
    args = argparse.Namespace(
        centers="10,20.5,31",
        center=None,
        spacing=None,
        n_traces=7,
    )

    np.testing.assert_allclose(cli._get_centers(args), [10.0, 20.5, 31.0])


def test_get_centers_builds_regular_bundle_about_center():
    args = argparse.Namespace(
        centers=None,
        center=100.0,
        spacing=10.0,
        n_traces=3,
    )

    np.testing.assert_allclose(cli._get_centers(args), [90.0, 100.0, 110.0])


def test_get_centers_rejects_mixed_geometry_forms():
    args = argparse.Namespace(
        centers="10,20",
        center=15.0,
        spacing=10.0,
        n_traces=2,
    )

    with pytest.raises(ValueError, match="either --centers or --center/--spacing"):
        cli._get_centers(args)


def test_get_centers_rejects_missing_geometry():
    args = argparse.Namespace(
        centers=None,
        center=15.0,
        spacing=None,
        n_traces=2,
    )

    with pytest.raises(ValueError, match="provide --centers or both"):
        cli._get_centers(args)


@pytest.mark.parametrize("centers", ["", "1,nan", "1,foo"])
def test_get_centers_rejects_invalid_explicit_centers(centers):
    args = argparse.Namespace(
        centers=centers,
        center=None,
        spacing=None,
        n_traces=2,
    )

    with pytest.raises(ValueError, match="trace centers"):
        cli._get_centers(args)


def test_parser_accepts_rebin_option():
    parser = cli._build_parser()

    args = parser.parse_args(
        [
            "l1",
            "input.fits",
            "output.fits",
            "--center",
            "100",
            "--spacing",
            "10",
            "--rebin",
            "4",
        ]
    )

    assert args.rebin == 4


def test_run_l1_passes_rebin_and_calibration_paths_to_writer(tmp_path, monkeypatch):
    input_path = tmp_path / "input.fits"
    output_path = tmp_path / "output.fits"
    bias_path = tmp_path / "bias.fits"
    fits.PrimaryHDU(np.ones((9, 20), dtype=float)).writeto(input_path)
    fits.PrimaryHDU(np.ones((9, 20), dtype=float)).writeto(bias_path)

    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "l1",
            str(input_path),
            str(output_path),
            "--centers",
            "4",
            "--half-width",
            "3",
            "--bias",
            str(bias_path),
            "--gain",
            "2",
            "--read-noise",
            "4",
            "--rebin",
            "5",
        ]
    )
    captured = {}

    def fake_process_l1(ccd, centers, **kwargs):
        captured["ccd"] = ccd
        captured["centers"] = np.asarray(centers)
        captured["process_kwargs"] = kwargs
        return ["spectrum"]

    def fake_write(path, spectra, centers, **kwargs):
        captured["write_path"] = path
        captured["spectra"] = spectra
        captured["write_centers"] = np.asarray(centers)
        captured["write_kwargs"] = kwargs

    monkeypatch.setattr(cli, "process_l1", fake_process_l1)
    monkeypatch.setattr(cli, "write_l1_fits", fake_write)

    cli._run_l1(args)

    assert captured["ccd"].unit == u.adu
    np.testing.assert_allclose(captured["centers"], [4.0])
    assert captured["process_kwargs"]["half_width"] == 3
    assert captured["process_kwargs"]["gain"] == 2.0
    assert captured["process_kwargs"]["read_noise"] == 4.0
    assert captured["write_path"] == output_path
    assert captured["write_kwargs"]["bias_path"] == bias_path
    assert captured["write_kwargs"]["rebin"] == 5


@pytest.mark.xfail(
    reason="Current HEAD parses --variance-ext/--mask-ext but does not load them into CCDData",
    strict=True,
)
def test_run_l1_loads_requested_variance_and_mask_extensions(tmp_path, monkeypatch):
    input_path = tmp_path / "input.fits"
    output_path = tmp_path / "output.fits"
    data = np.ones((9, 20), dtype=float)
    variance = np.full_like(data, 7.0)
    mask = np.zeros_like(data, dtype=np.uint8)
    mask[4, 10] = 1
    fits.HDUList(
        [
            fits.PrimaryHDU(data),
            fits.ImageHDU(variance, name="VARIANCE"),
            fits.ImageHDU(mask, name="MASK"),
        ]
    ).writeto(input_path)

    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "l1",
            str(input_path),
            str(output_path),
            "--centers",
            "4",
            "--half-width",
            "3",
            "--variance-ext",
            "VARIANCE",
            "--mask-ext",
            "MASK",
        ]
    )
    captured = {}

    def fake_process_l1(ccd, centers, **kwargs):
        captured["ccd"] = ccd
        return []

    monkeypatch.setattr(cli, "process_l1", fake_process_l1)
    monkeypatch.setattr(cli, "write_l1_fits", lambda *args, **kwargs: None)

    cli._run_l1(args)

    assert isinstance(captured["ccd"].uncertainty, VarianceUncertainty)
    np.testing.assert_allclose(captured["ccd"].uncertainty.array, variance)
    np.testing.assert_array_equal(captured["ccd"].mask, mask.astype(bool))
