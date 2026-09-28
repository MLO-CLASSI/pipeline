from pipeline.l2 import process_l2


def test_l2_remains_explicitly_unimplemented():
    assert process_l2("input.fits", "output.fits") is NotImplemented
