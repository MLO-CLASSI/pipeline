# Spectrograph pipeline

Initial CLI scaffold for the fiber-fed spectrograph. The current scope is only
Horne optimal extraction of known, flat traces using `specreduce`.

## Assumptions in this first version

- The input is a 2-D FITS image.
- Dispersion is along FITS/Numpy axis 1 (horizontal/X).
- Fiber traces are flat and their Y centers are supplied by the user.
- Bias/background/flat processing is intentionally not implemented yet. The
  input should already be background-subtracted enough for optimal extraction.
- A variance plane should ideally be supplied. As a fallback, the CLI can build
  a simple Poisson + read-noise variance estimate from gain and read noise.
- Each fiber is extracted from its own local cross-dispersion cutout so that the
  other six traces do not contaminate `specreduce`'s spatial-profile fit.
- The default cutout half-width is inferred from half the minimum fiber spacing;
  it can be overridden with `--half-width`.
- Fiber cross-talk is not modeled. Each trace is extracted independently with a
  Gaussian spatial profile and a fixed zero background term.

## Install

```bash
python -m pip install -e .
```

## Usage

With a variance extension:

```bash
spectrograph-pipeline extract science.fits extracted.fits \
    --variance-ext VARIANCE \
    --center 512.0 \
    --spacing 25.7
```

Or with explicit trace centers:

```bash
spectrograph-pipeline extract science.fits extracted.fits \
    --variance-ext 1 \
    --centers 434.9,460.6,486.3,512.0,537.7,563.4,589.1
```

If no variance extension exists, a provisional variance model can be used:

```bash
spectrograph-pipeline extract science.fits extracted.fits \
    --center 512.0 \
    --spacing 25.7 \
    --gain 1.2 \
    --read-noise 3.5
```

The output is a FITS file with one binary-table extension per fiber. Each table
contains `PIXEL`, `FLUX`, `SIGMA`, and `MASK`. Wavelength calibration is
intentionally deferred, so the spectral coordinate is detector pixel number.
