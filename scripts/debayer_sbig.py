
import argparse
import ccdproc
import cv2
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.nddata import CCDData


def debayer_image(file: Path) -> CCDData:
    raw = CCDData.read(file, unit="adu")
    saturated = raw.data >= 63000
    bayer = np.round(raw.data-raw.data.min()).astype(np.uint16)
    rgb = np.moveaxis(cv2.cvtColor(bayer, cv2.COLOR_BAYER_GBRG2RGB), -1, 0)
    new = CCDData(rgb, unit="adu")
    new.header = raw.header
    return new


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)

    args = parser.parse_args()
    debayered = debayer_image(args.input_file)
    debayered.write(args.input_file.with_suffix(".rgb.fits"), overwrite=True)
