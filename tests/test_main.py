import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import CCDData, VarianceUncertainty

import pipeline.l1.main as l1_main
from pipeline.l1.main import apply_l1_corrections, process_l1


def test_apply_l1_corrections_returns_copy_when_no_calibrations_are_supplied():
    ccd = CCDData(np.arange(4, dtype=float).reshape(2, 2), unit=u.adu)

    result = apply_l1_corrections(ccd)

    assert result is not ccd
    np.testing.assert_array_equal(result.data, ccd.data)


@pytest.mark.parametrize("calibration_name", ["bias", "dark", "flat"])
def test_apply_l1_corrections_rejects_mismatched_calibration_shape(calibration_name):
    ccd = CCDData(np.ones((4, 4)), unit=u.adu)
    calibration = CCDData(np.ones((3, 4)), unit=u.adu)

    kwargs = {calibration_name: calibration}
    with pytest.raises(ValueError, match=rf"master {calibration_name} shape"):
        apply_l1_corrections(ccd, **kwargs)


def test_apply_l1_corrections_bias_dark_and_constant_flat():
    ccd = CCDData(np.full((2, 2), 20.0), unit=u.adu)
    bias = CCDData(np.full((2, 2), 2.0), unit=u.adu)
    dark = CCDData(np.full((2, 2), 3.0), unit=u.adu)
    flat = CCDData(np.full((2, 2), 5.0), unit=u.adu)

    result = apply_l1_corrections(ccd, bias=bias, dark=dark, flat=flat)

    np.testing.assert_allclose(result.data, 15.0)
    np.testing.assert_allclose(ccd.data, 20.0)


def test_apply_l1_corrections_scales_dark_by_exposure_time():
    ccd = CCDData(np.full((2, 2), 10.0), unit=u.adu, meta={"EXPTIME": 10.0})
    dark = CCDData(np.full((2, 2), 2.0), unit=u.adu, meta={"EXPTIME": 5.0})

    result = apply_l1_corrections(ccd, dark=dark, dark_scale=True)

    np.testing.assert_allclose(result.data, 6.0)


def test_process_l1_passes_variance_mask_unit_and_explicit_half_width_to_extraction(monkeypatch):
    data = np.full((9, 20), 12.0)
    mask = np.zeros_like(data, dtype=bool)
    mask[4, 5] = True
    ccd = CCDData(data, unit=u.adu, mask=mask)
    captured = {}

    def fake_extract(data, variance, centers, half_width, mask, unit):
        captured["data"] = data
        captured["variance"] = variance
        captured["centers"] = list(centers)
        captured["half_width"] = half_width
        captured["mask"] = mask
        captured["unit"] = unit
        return ["spectrum"]

    monkeypatch.setattr(l1_main, "boxcar_extract_traces", fake_extract)

    result = process_l1(
        ccd,
        centers=[4.0],
        half_width=3,
        gain=2.0,
        read_noise=4.0,
    )

    assert result == ["spectrum"]
    np.testing.assert_allclose(captured["data"], data)
    np.testing.assert_allclose(captured["variance"], 10.0)
    np.testing.assert_array_equal(captured["mask"], mask)
    assert captured["centers"] == [4.0]
    assert captured["half_width"] == 3
    assert captured["unit"] == u.adu


def test_process_l1_infers_half_width(monkeypatch):
    ccd = CCDData(
        np.ones((20, 20)),
        unit=u.adu,
        uncertainty=VarianceUncertainty(np.ones((20, 20))),
    )
    captured = {}

    def fake_extract(data, variance, centers, half_width, mask, unit):
        captured["half_width"] = half_width
        return []

    monkeypatch.setattr(l1_main, "boxcar_extract_traces", fake_extract)

    process_l1(ccd, centers=[5.0, 12.0])

    assert captured["half_width"] == 3


def test_process_l1_rejects_missing_uncertainty_after_corrections(monkeypatch):
    ccd = CCDData(
        np.ones((5, 10)),
        unit=u.adu,
        uncertainty=VarianceUncertainty(np.ones((5, 10))),
    )

    monkeypatch.setattr(
        l1_main,
        "apply_l1_corrections",
        lambda *args, **kwargs: CCDData(np.ones((5, 10)), unit=u.adu),
    )

    with pytest.raises(ValueError, match="produced an image with no uncertainty"):
        process_l1(ccd, centers=[2.0], half_width=2)
