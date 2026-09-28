import numpy as np
from astropy import units as u
from astropy.nddata import CCDData

import pipeline.l1.cosmic as cosmic


def test_apply_cosmicray_correction_uses_configured_lacosmic_defaults(monkeypatch):
    ccd = CCDData(np.ones((5, 5)), unit=u.adu)
    expected = CCDData(np.zeros((5, 5)), unit=u.adu)
    captured = {}

    def fake_lacosmic(data, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(cosmic, "cosmicray_lacosmic", fake_lacosmic)

    result = cosmic.apply_cosmicray_correction(ccd)

    assert result is expected
    assert captured["data"] is ccd
    assert captured["kwargs"] == {
        "sigclip": 5.0,
        "gain": 1.0,
        "readnoise": 1.0,
    }
