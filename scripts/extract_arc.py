"""
First pass extraction of an MLO fiber-fed spectrograph arc lamp exposure.

Reads instrument metadata straight from the FITS header (populated by the
Flask ICS), traces the fiber's near field image with a FlatTrace (the trace
is empirically flat across this frame --> see diagnostic plot), subtracts a
local background, and extracts a 1D spectrum with both Boxcar and Horne
(optimal) methods for comparison.

"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import CCDData, VarianceUncertainty
import astropy.units as u

from specreduce.tracing import FlatTrace
from specreduce.background import Background
from specreduce.extract import BoxcarExtract, HorneExtract

# Load data + pull instrument metadata from the header
FITS_PATH = "data/20260814T140330_light_argon_benchtest.fits"
hdul = fits.open(FITS_PATH)
hdr = hdul[0].header
data = hdul[0].data.astype(float)

instrument_info = {
    "telescope": hdr.get("TELESCOP"),
    "instrument": hdr.get("INSTRUME"),
    "camera": hdr.get("CAMERA"),
    "filter": hdr.get("FILTER"),
    "grating": hdr.get("GRATING"),
    "object": hdr.get("OBJECT"),
    "exptime_s": hdr.get("EXPTIME"),
    "camfocus": hdr.get("CAMFOCUS"),
    "ccd_temp_C": hdr.get("CCD-TEMP"),
    "date_obs": hdr.get("DATE-OBS"),
}
print("Instrument config (from ICS FITS header):")
for k, v in instrument_info.items():
    print(f"  {k:12s}: {v}")

# Build a simple read noise + shot-noise variance estimate 
GAIN = 0.37          # e-/ADU  (placeholder from GSENSE400BI sensor -->  replace with measured value)
READ_NOISE = 9.3      # e- (placeholder from GSENSE400BI sensor --> replace with measured value)

signal_e = np.clip(data, 0, None) * GAIN
variance_adu = (signal_e + READ_NOISE**2) / GAIN**2  # back to ADU^2

ccd = CCDData(data, unit=u.adu, uncertainty=VarianceUncertainty(variance_adu, unit=u.adu**2))

# Trace: flat, based on the diagnostic 
trace_pos = 1076  # from the Gaussian fits done earlier
trace = FlatTrace(ccd, trace_pos=trace_pos)

# Background subtraction: two-sided windows away from the trace 
bg = Background.two_sided(ccd, trace, separation=25, width=15)
ccd_bgsub = ccd.data - bg.bkg_image().data

ccd_bgsub_nd = CCDData(ccd_bgsub, unit=u.adu, uncertainty=ccd.uncertainty)

# Extract: Boxcar (simple, fast) and Horne (optimal, variance-weighted)
boxcar = BoxcarExtract(ccd_bgsub_nd, trace, width=15)
spec_boxcar = boxcar.spectrum

horne = HorneExtract(ccd_bgsub_nd, trace)
spec_horne = horne.spectrum
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(spec_boxcar.spectral_axis, spec_boxcar.flux, lw=0.7, color='tab:blue')
axes[0].set_title(f"Simplified extraction (boxcar) (width=15 px) -- {instrument_info['object']}, "
                   f"{instrument_info['grating']}")
axes[0].set_ylabel("Flux [ADU]")

axes[1].plot(spec_horne.spectral_axis, spec_horne.flux, lw=0.7, color='tab:orange')
axes[1].set_title("Optimal (Horne) extraction")
axes[1].set_ylabel("Flux [ADU]")
axes[1].set_xlabel("Dispersion axis [pixel]")

plt.tight_layout()
plt.savefig("outputs/extracted_spectrum_comparison.png", dpi=130)
print("\nSaved comparison plot.")

np.savetxt(
    "outputs/extracted_simple.csv",
    np.column_stack([spec_boxcar.spectral_axis.value, spec_boxcar.flux.value]),
    delimiter=",", header="pixel,flux_adu", comments=""
)
np.savetxt(
    "outputs/extracted_optimal.csv",
    np.column_stack([spec_horne.spectral_axis.value, spec_horne.flux.value]),
    delimiter=",", header="pixel,flux_adu", comments=""
)
print("Saved extracted_simple.csv and extracted_optimal.csv")
