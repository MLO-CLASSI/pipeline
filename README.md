# CLASSI Spectrograph pipeline

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

#### Baseline wavelength solution

The baseline wavelength calibration is a versioned ASDF reference product
derived from arc-lamp exposures. One file contains the solutions for all fiber
traces. The actual pixel-to-wavelength transforms are stored as GWCS objects,
using the same lossless representation used by `specreduce`'s
`WavelengthSolution1D` rather than a CLASSI-specific coefficient encoding.
CLASSI's wrapper adds only multi-trace organization, arc-line diagnostics, and
provenance.

The top-level `classi_wavesol` tree records the format/version, wavelength unit
and medium, native pixel-coordinate convention, arc exposure and lamps, and a
list of traces. Each trace contains:

- a `wavelength_solution` node with the GWCS transform and its native pixel
  bounding box;
- the matched arc-line pixel positions and reference wavelengths;
- residuals and the used/rejected flag for each line;
- line identifiers and RMS fit diagnostics.

Pixel coordinates are always native zero-based detector coordinates. This
remains true when an L1 spectrum has been rebinned, because its `PIXEL` column
retains native-detector coordinates. Internal wavelengths should normally be
vacuum wavelengths; the ASDF metadata records the medium explicitly.

`pipeline.l2.fit_arc_wavelength_solution()` uses
`specreduce.wavecal1d.WavelengthCalibration1D.fit_lines()` for the arc fit.
`write_wavesol()` and `read_wavesol()` handle the CLASSI multi-trace ASDF
container. The final L2 science product can remain FITS; the ASDF file is the
long-lived calibration/reference object.

#### Sky-line refinement

The arc solution defines the detailed dispersion relation. For an individual
science exposure, L2 can refine it by fitting an affine detector-coordinate
transform to measured night-sky line centroids:

```text
x_master = REFPIX + dx + scale * (x_science - REFPIX)
lambda_science(x) = lambda_master(x_master)
```

`pipeline.l2.fit_sky_refinement()` implements this fit. With one usable sky
line, `mode="auto"` fits only `dx`; with two or more it fits both `dx` and the
stretch `scale`. The resulting transform is composed exactly with the master
polynomial rather than refitting its higher-order shape. Automatic selection
and centroiding of sky lines is intentionally kept as a separate L2 step so
the pipeline can choose the cleanest available sky spectrum before fitting the
refinement.

### L3 - target photometric anchoring

L3 is reserved for target-specific calibration that uses external observations
rather than instrument calibration data. The initial L3 implementation provides
a broadband photometric anchoring model for an already wavelength- and
flux-calibrated spectrum. It fits a positive multiplicative correction

```text
C(lambda) = exp(a0 + a1 log(lambda/lambda_ref) + a2 log(lambda/lambda_ref)^2)
```

against synthetic photometry through the full filter bandpasses. One band fits
a gray scale factor, two bands fit scale plus color, and three or more bands fit
up to quadratic curvature by default. This is intended for nearly simultaneous
CLASSI target photometry, typically B/V/R. The built-in aliases use Johnson B,
Johnson V, and Cousins R; arbitrary `synphot` bandpasses can also be supplied.

The L3 numerical machinery is implemented independently of FITS I/O for now,
because the L2 file format is not yet defined. The fitted coefficient covariance
is retained separately from the per-pixel statistical uncertainty because the
photometric-calibration error is correlated across wavelength.

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

The installed import namespace is `pipeline`, and installation provides the
`classi-pipeline` command.

## L1 usage

With no detector corrections:

```bash
classi-pipeline l1 science_l0.fits science_l1.fits \
    --center 1023.5 \
    --spacing 50
```

With explicit trace centers and detector calibration frames:

```bash
classi-pipeline l1 science_l0.fits science_l1.fits \
    --bias master_bias.fits \
    --dark master_dark.fits \
    --dark-scale \
    --flat master_flat.fits \
    --centers 434.9,460.6,486.3,512.0,537.7,563.4,589.1
```

A provisional variance model can be initialized from gain and read noise:

```bash
classi-pipeline l1 science_l0.fits science_l1.fits \
    --center 512.0 \
    --spacing 25.7 \
    --gain 1.2 \
    --read-noise 3.5
```

In the future, the L2 interface will be called as:

```bash
classi-pipeline l2 science_l1.fits science_l2.fits
```
