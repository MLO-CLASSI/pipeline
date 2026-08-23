# Spectrograph pipeline

## Data levels

### L0 - "raw" ICS product

L0 is produced by the instrument-control system in essentially real time. This
pipeline does not create L0 products. An L0 product is the detector FITS image
with the instrument/observation metadata injected by the ICS, but without any
science-pipeline processing.

### L1 - detector processing and extraction

L1 consumes an L0 FITS image and produces one-dimensional spectra in detector
pixel coordinates. It currently supports:

- optional bias subtraction
- optional dark subtraction
- optional flat correction
- propagated image uncertainty through those corrections using `CCDData` and
  `ccdproc`
- optimal extraction of traces using `specreduce`

L1 does **not** perform wavelength calibration or spectrophotometric calibration.

Each extracted trace is written as a binary table HDU with the columns:

- `PIXEL` — zero-based detector dispersion coordinate;
- `COUNTS` — extracted detector-domain signal;
- `SIGMA` — 1-sigma uncertainty in the same unit as `COUNTS`;
- `MASK` — invalid/bad spectral samples.

The primary header records `PROCLVL=1`, which detector corrections were
applied, and explicitly records `WAVECAL=F` and `FLUXCAL=F`.

### L2 - physical spectral calibration

L2 will consume L1 products and be responsible for the physical calibration of
the spectra, including:

- wavelength solution and wavelength-coordinate assignment;
- instrumental response / sensitivity correction;
- atmospheric-extinction correction as appropriate;
- spectrophotometric flux calibration;
- later calibration-related operations such as telluric treatment if desired.

The L2 function is stubbed out in the framework but is not implemented yet.
A future L2 table can retain `PIXEL` for provenance while adding a physical
`WAVELENGTH` coordinate and calibrated `FLUX`/uncertainty columns.

## Current L1 assumptions

- Dispersion is along array axis 1 (horizontal/X).
- Traces are currently flat and their Y centers are supplied explicitly or as a bundle center plus regular spacing.
- Seven traces are the default, but the count is configurable with `--n-traces`.
- Each trace is extracted from its own local cross-dispersion cutout so nearby
  traces do not contaminate `specreduce`'s spatial-profile fit.
- Trace cross-talk is not modeled. Each trace is extracted independently with a
  Gaussian spatial profile and a fixed zero background term.
- Calibration images are assumed to already be master products. A flat supplied here should represent detector/pixel-response correction rather than a spectrophotometric response function. Master-frame uncertainty is not (yet) read or propagated.
- A dark used with `--dark-scale` should already be bias-subtracted and must carry the exposure-time keyword specified by `--exposure-key`.

## Install

```bash
python -m pip install -e .
```

## L1 usage

With no detector corrections:

```bash
python pipeline l1 science_l0.fits science_l1.fits \
    --center 1023.5 \
    --spacing 50
```

With explicit trace centers and detector calibration frames:

```bash
python pipeline l1 science_l0.fits science_l1.fits \
    --bias master_bias.fits \
    --dark master_dark.fits \
    --dark-scale \
    --flat master_flat.fits \
    --centers 434.9,460.6,486.3,512.0,537.7,563.4,589.1
```

A provisional variance model can be initialized from gain and read noise:

```bash
python pipeline l1 science_l0.fits science_l1.fits \
    --center 512.0 \
    --spacing 25.7 \
    --gain 1.2 \
    --read-noise 3.5
```

In the future, the L2 interface will be called as:

```bash
python pipeline l2 science_l1.fits science_l2.fits
```
