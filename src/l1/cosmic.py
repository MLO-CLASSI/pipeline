
from astropy.nddata import CCDData
from ccdproc import cosmicray_lacosmic

def apply_cosmicray_correction(data: CCDData):
    ccd = cosmicray_lacosmic(
        data,
        sigclip=5.0,
        gain=1.0,
        readnoise=1.0,
    )
    return ccd
