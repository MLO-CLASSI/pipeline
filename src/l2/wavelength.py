"""Wavelength-calibration support for L2.

The baseline CLASSI wavelength calibration is derived from arc-lamp spectra and
stored as a multi-trace ASDF ``wavesol`` reference file. Each trace stores the
same GWCS pixel-to-wavelength transform used by ``specreduce``'s
``WavelengthSolution1D`` rather than a CLASSI-specific coefficient encoding.

Per-observation sky-line refinement is represented as an affine transform of
science-frame pixel coordinates before evaluating the baseline solution::

    x_master = x_ref + dx + scale * (x_science - x_ref)
    wavelength = lambda_master(x_master)

This preserves the detailed arc-derived dispersion relation while allowing the
science exposure to correct a zero-point shift and, when the sky lines support
it, a small dispersion stretch.
"""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import asdf
import numpy as np
from astropy import units as u
from astropy.modeling import CompoundModel, models
from astropy.time import Time
from specreduce.wavecal1d import WavelengthCalibration1D
from specreduce.wavesol1d import WavelengthSolution1D


WAVESOL_FORMAT = "CLASSI-WAVESOL"
WAVESOL_VERSION = 1


@dataclass(frozen=True)
class ArcWavelengthSolution:
    """One trace's arc-derived wavelength solution and fit diagnostics."""

    trace_id: int
    solution: WavelengthSolution1D
    line_pixels: np.ndarray
    line_wavelengths: u.Quantity
    used: np.ndarray
    line_ids: tuple[str, ...]

    @property
    def reference_pixel(self) -> float:
        """Detector pixel about which the polynomial is defined."""
        return float(-self.solution.p2w[0].offset.value)

    @property
    def wavelength_residuals(self) -> u.Quantity:
        """Catalog minus fitted wavelength for each stored line."""
        fitted = self.solution.pix_to_wav(self.line_pixels) * self.solution.unit
        return self.line_wavelengths - fitted

    @property
    def pixel_residuals(self) -> np.ndarray:
        """Observed minus fitted pixel coordinate for each stored line."""
        fitted = self.solution.wav_to_pix(self.line_wavelengths.to_value(self.solution.unit))
        return self.line_pixels - fitted

    @property
    def rms_wavelength(self) -> u.Quantity:
        residuals = self.wavelength_residuals[self.used]
        if residuals.size == 0:
            return np.nan * self.solution.unit
        return np.sqrt(np.mean(residuals**2))

    @property
    def rms_pixel(self) -> float:
        residuals = self.pixel_residuals[self.used]
        if residuals.size == 0:
            return np.nan
        return float(np.sqrt(np.mean(residuals**2)))


@dataclass(frozen=True)
class SkyRefinement:
    """Affine pixel-coordinate refinement derived from night-sky lines."""

    mode: str
    reference_pixel: float
    shift: float
    stretch: float
    rms_pixel: float
    n_lines: int
    covariance: np.ndarray | None
    observed_pixels: np.ndarray
    master_pixels: np.ndarray
    wavelengths: u.Quantity

    def master_pixel(self, pixel) -> np.ndarray:
        """Map science-frame pixel coordinates onto the master-arc coordinate system."""
        pixel = np.asarray(pixel, dtype=float)
        return (
            self.reference_pixel
            + self.shift
            + self.stretch * (pixel - self.reference_pixel)
        )

    def refined_solution(self, baseline: WavelengthSolution1D) -> WavelengthSolution1D:
        """Return the exact polynomial produced by composing this affine correction."""
        if baseline.p2w is None:
            raise ValueError("baseline wavelength solution is not set")
        baseline_reference = float(-baseline.p2w[0].offset.value)
        if not np.isclose(baseline_reference, self.reference_pixel):
            raise ValueError("sky refinement reference pixel does not match the baseline solution")

        polynomial = baseline.p2w[1]
        coefficients = np.array(
            [getattr(polynomial, f"c{i}").value for i in range(polynomial.degree + 1)],
            dtype=float,
        )
        base = np.polynomial.Polynomial(coefficients)
        affine = np.polynomial.Polynomial([self.shift, self.stretch])
        refined = base(affine)

        model = models.Shift(-self.reference_pixel) | models.Polynomial1D(
            polynomial.degree,
            **{f"c{i}": coefficient for i, coefficient in enumerate(refined.coef)},
        )
        return _new_wavelength_solution(
            model,
            baseline.bounds_pix,
            baseline.unit,
            wave_air=_wave_air(baseline),
        )


def _wave_air(solution: WavelengthSolution1D) -> bool:
    """Return the solution wavelength medium across supported specreduce versions."""
    return bool(getattr(solution, "wave_air", False))


def _new_wavelength_solution(
    p2w: CompoundModel,
    bounds_pix: tuple[int, int],
    unit: u.UnitBase,
    *,
    wave_air: bool,
) -> WavelengthSolution1D:
    """Construct a wavelength solution across specreduce 1.9/1.10 API differences."""
    try:
        solution = WavelengthSolution1D(p2w, bounds_pix, unit, wave_air=wave_air)
    except TypeError as exc:
        if "wave_air" not in str(exc):
            raise
        solution = WavelengthSolution1D(p2w, bounds_pix, unit)
        solution.wave_air = bool(wave_air)
    return solution


def _solution_asdf_node(solution: WavelengthSolution1D) -> dict:
    """Return the lossless GWCS representation used by specreduce ASDF serialization."""
    if solution.p2w is None:
        raise ValueError("wavelength solution is not set")

    wcs = deepcopy(solution.gwcs)
    wcs.bounding_box = (solution.bounds_pix,)
    return {
        "gwcs": wcs,
        "wave_air": _wave_air(solution),
    }


def _solution_from_asdf_node(node) -> WavelengthSolution1D:
    """Restore a ``WavelengthSolution1D`` from its stored GWCS representation."""
    try:
        wcs = node["gwcs"]
        wave_air = bool(node["wave_air"])
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid wavelength_solution node") from exc

    if wcs.bounding_box is None:
        raise ValueError("stored wavelength solution has no pixel bounding box")
    bounds_pix = wcs.bounding_box.bounding_box()
    unit = u.Unit(wcs.output_frame.unit[0])
    p2w = wcs.forward_transform.copy()

    if hasattr(p2w, "bounding_box"):
        del p2w.bounding_box
    if not (
        isinstance(p2w, CompoundModel)
        and isinstance(p2w[0], models.Shift)
        and isinstance(p2w[1], models.Polynomial1D)
    ):
        raise ValueError("stored wavelength solution is not a shift followed by a polynomial")

    return _new_wavelength_solution(
        p2w,
        (int(bounds_pix[0]), int(bounds_pix[1])),
        unit,
        wave_air=wave_air,
    )


def fit_arc_wavelength_solution(
    trace_id: int,
    pixels,
    wavelengths,
    pix_bounds: tuple[int, int],
    *,
    degree: int = 2,
    unit: u.UnitBase = u.AA,
    ref_pixel: float | None = None,
    wave_air: bool = False,
    line_ids: Sequence[str] | None = None,
    used: Sequence[bool] | None = None,
) -> ArcWavelengthSolution:
    """Fit a trace's baseline wavelength solution from matched arc lines.

    ``pix_bounds`` uses Python-style bounds: the lower bound is inclusive and
    the upper bound is exclusive. For an unbinned 4096-pixel detector axis this
    would normally be ``(0, 4096)``.
    """
    pixels = np.asarray(pixels, dtype=float)
    if isinstance(wavelengths, u.Quantity):
        wavelengths = wavelengths.to(unit)
    else:
        wavelengths = np.asarray(wavelengths, dtype=float) * unit

    if pixels.ndim != 1 or wavelengths.ndim != 1 or pixels.size != wavelengths.size:
        raise ValueError("pixels and wavelengths must be matching one-dimensional arrays")
    if degree < 1:
        raise ValueError("wavelength-solution degree must be at least 1")
    if pix_bounds[1] <= pix_bounds[0]:
        raise ValueError("pix_bounds must be an increasing (min, max) pair")
    if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(wavelengths.value)):
        raise ValueError("arc-line pixels and wavelengths must be finite")

    if ref_pixel is None:
        ref_pixel = (pix_bounds[0] + pix_bounds[1] - 1) / 2

    if line_ids is None:
        line_ids = ("",) * pixels.size
    else:
        line_ids = tuple(str(value) for value in line_ids)
        if len(line_ids) != pixels.size:
            raise ValueError("line_ids must have one entry per matched arc line")

    if used is None:
        used = np.ones(pixels.size, dtype=bool)
    else:
        used = np.asarray(used, dtype=bool)
        if used.shape != pixels.shape:
            raise ValueError("used must have one entry per matched arc line")
    if np.count_nonzero(used) < degree + 1:
        raise ValueError("at least degree + 1 used arc lines are required")

    calibration = WavelengthCalibration1D(
        unit=unit,
        ref_pixel=float(ref_pixel),
        pix_bounds=pix_bounds,
        wave_air=wave_air,
    )
    solution = calibration.fit_lines(
        pixels[used],
        wavelengths[used].to_value(unit),
        degree=degree,
        refine_fit=False,
    )
    # specreduce 1.9 accepts wave_air on WavelengthCalibration1D but does not
    # propagate it onto WavelengthSolution1D; 1.10 does. Preserve it explicitly.
    solution.wave_air = bool(wave_air)

    return ArcWavelengthSolution(
        trace_id=int(trace_id),
        solution=solution,
        line_pixels=pixels,
        line_wavelengths=wavelengths,
        used=np.asarray(used, dtype=bool),
        line_ids=tuple(line_ids),
    )


def write_wavesol(
    path: str | Path,
    solutions: Sequence[ArcWavelengthSolution],
    *,
    arc_file: str | Path | None = None,
    lamps: Sequence[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Write a versioned multi-trace CLASSI wavelength reference as ASDF.

    The actual pixel-to-wavelength transforms are stored as GWCS objects, using
    the same lossless representation adopted by ``WavelengthSolution1D`` for
    upstream specreduce ASDF serialization. CLASSI adds only the multi-trace
    wrapper and arc-line diagnostics/provenance.
    """
    solutions = tuple(solutions)
    if not solutions:
        raise ValueError("at least one trace wavelength solution is required")

    trace_ids = [item.trace_id for item in solutions]
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("trace IDs in a wavesol file must be unique")

    unit = solutions[0].solution.unit
    wave_air = _wave_air(solutions[0].solution)
    for item in solutions[1:]:
        if item.solution.unit != unit:
            raise ValueError("all trace solutions in a wavesol file must use the same unit")
        if _wave_air(item.solution) != wave_air:
            raise ValueError("all trace solutions must use the same wavelength medium")

    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output wavesol already exists: {path}")
        path.unlink()

    traces = []
    for item in sorted(solutions, key=lambda value: value.trace_id):
        traces.append(
            {
                "trace_id": int(item.trace_id),
                "wavelength_solution": _solution_asdf_node(item.solution),
                "fit": {
                    "rms_wavelength": item.rms_wavelength,
                    "rms_pixel": float(item.rms_pixel),
                    "n_used": int(np.count_nonzero(item.used)),
                },
                "lines": {
                    "pixel": np.asarray(item.line_pixels, dtype=float),
                    "wavelength": item.line_wavelengths,
                    "wavelength_residual": item.wavelength_residuals,
                    "pixel_residual": np.asarray(item.pixel_residuals, dtype=float),
                    "used": np.asarray(item.used, dtype=bool),
                    "line_id": list(item.line_ids),
                },
            }
        )

    tree = {
        "classi_wavesol": {
            "format": WAVESOL_FORMAT,
            "version": WAVESOL_VERSION,
            "calibration_type": "arc",
            "created": Time.now().isot,
            "n_traces": len(traces),
            "pixel_origin": 0,
            "dispersion_axis": 1,
            "wavelength_unit": unit.to_string("fits"),
            "wavelength_medium": "air" if wave_air else "vacuum",
            "arc_file": None if arc_file is None else str(arc_file),
            "lamps": [] if lamps is None else [str(lamp) for lamp in lamps],
            "traces": traces,
        }
    }
    asdf.AsdfFile(tree).write_to(path)


def read_wavesol(path: str | Path) -> dict[int, ArcWavelengthSolution]:
    """Read a CLASSI baseline wavelength-solution ASDF file."""
    with asdf.open(path, lazy_load=False) as af:
        try:
            root = af["classi_wavesol"]
        except KeyError as exc:
            raise ValueError(f"{path} contains no CLASSI wavesol tree") from exc

        if root.get("format") != WAVESOL_FORMAT:
            raise ValueError(f"{path} is not a {WAVESOL_FORMAT} file")
        if root.get("version") != WAVESOL_VERSION:
            raise ValueError(f"unsupported wavesol format version {root.get('version')!r}")
        if root.get("pixel_origin") != 0:
            raise ValueError("wavesol pixel_origin must be zero")
        if root.get("dispersion_axis") != 1:
            raise ValueError("wavesol dispersion_axis must be 1")

        unit = u.Unit(root["wavelength_unit"])
        medium = str(root["wavelength_medium"]).lower()
        if medium not in {"air", "vacuum"}:
            raise ValueError(f"unsupported wavelength medium {medium!r}")
        wave_air = medium == "air"

        result = {}
        for trace in root["traces"]:
            trace_id = int(trace["trace_id"])
            if trace_id in result:
                raise ValueError(f"duplicate trace ID {trace_id} in wavesol")

            solution = _solution_from_asdf_node(trace["wavelength_solution"])
            if solution.unit != unit:
                raise ValueError(f"trace {trace_id} wavelength unit does not match file metadata")
            if _wave_air(solution) != wave_air:
                raise ValueError(f"trace {trace_id} wavelength medium does not match file metadata")

            lines = trace["lines"]
            line_pixels = np.array(lines["pixel"], dtype=float, copy=True)
            line_wavelengths = u.Quantity(lines["wavelength"]).to(unit).copy()
            used = np.array(lines["used"], dtype=bool, copy=True)
            line_ids = tuple(str(value) for value in lines["line_id"])
            if not (
                line_pixels.ndim == 1
                and line_wavelengths.ndim == 1
                and used.ndim == 1
                and line_pixels.size == line_wavelengths.size == used.size == len(line_ids)
            ):
                raise ValueError(f"trace {trace_id} has inconsistent arc-line arrays")

            result[trace_id] = ArcWavelengthSolution(
                trace_id=trace_id,
                solution=solution,
                line_pixels=line_pixels,
                line_wavelengths=line_wavelengths,
                used=used,
                line_ids=line_ids,
            )

        if len(result) != int(root["n_traces"]):
            raise ValueError("wavesol n_traces does not match the number of trace records")
        return result


def fit_sky_refinement(
    baseline: WavelengthSolution1D,
    observed_pixels,
    wavelengths,
    *,
    pixel_uncertainty=None,
    mode: str = "auto",
) -> SkyRefinement:
    """Fit a shift or affine stretch relative to an arc-derived master solution.

    The caller supplies measured sky-line centroids in the science exposure and
    their known wavelengths. Automatic sky-line detection/centroiding is kept
    separate from this fit so that L2 can choose the cleanest available sky
    spectrum (for example, a dedicated sky fiber or a combination of fibers).

    ``mode='auto'`` fits a shift from one line and an affine shift+stretch from
    two or more lines.
    """
    observed_pixels = np.asarray(observed_pixels, dtype=float)
    if isinstance(wavelengths, u.Quantity):
        wavelengths = wavelengths.to(baseline.unit)
    else:
        wavelengths = np.asarray(wavelengths, dtype=float) * baseline.unit

    if observed_pixels.ndim != 1 or wavelengths.ndim != 1:
        raise ValueError("observed pixels and sky wavelengths must be one-dimensional")
    if observed_pixels.size != wavelengths.size or observed_pixels.size == 0:
        raise ValueError("observed pixels and sky wavelengths must have matching nonzero size")
    if not np.all(np.isfinite(observed_pixels)) or not np.all(np.isfinite(wavelengths.value)):
        raise ValueError("sky-line pixels and wavelengths must be finite")

    if pixel_uncertainty is not None:
        pixel_uncertainty = np.asarray(pixel_uncertainty, dtype=float)
        if pixel_uncertainty.shape != observed_pixels.shape:
            raise ValueError("pixel_uncertainty must match observed_pixels")
        if np.any(~np.isfinite(pixel_uncertainty)) or np.any(pixel_uncertainty <= 0):
            raise ValueError("pixel uncertainties must be finite and positive")

    master_pixels = np.asarray(
        baseline.wav_to_pix(wavelengths.to_value(baseline.unit)),
        dtype=float,
    )
    lower_bound, upper_bound = baseline.bounds_pix
    if not np.all(np.isfinite(master_pixels)) or np.any(
        (master_pixels < lower_bound) | (master_pixels >= upper_bound)
    ):
        raise ValueError("one or more sky wavelengths lie outside the baseline solution")

    mode = mode.lower()
    if mode == "auto":
        mode = "affine" if observed_pixels.size >= 2 else "shift"
    if mode not in {"shift", "affine"}:
        raise ValueError("sky-refinement mode must be 'auto', 'shift', or 'affine'")
    if mode == "affine" and observed_pixels.size < 2:
        raise ValueError("affine sky refinement requires at least two sky lines")

    reference_pixel = float(-baseline.p2w[0].offset.value)
    y = master_pixels - reference_pixel
    x = observed_pixels - reference_pixel

    if mode == "shift":
        offsets = y - x
        if pixel_uncertainty is None:
            shift = float(np.mean(offsets))
            covariance = None
        else:
            weights = 1.0 / pixel_uncertainty**2
            shift = float(np.average(offsets, weights=weights))
            covariance = np.array([[1.0 / np.sum(weights)]])
        stretch = 1.0
        residuals = offsets - shift
    else:
        design = np.column_stack([np.ones(observed_pixels.size), x])
        if pixel_uncertainty is None:
            fit_design = design
            fit_y = y
        else:
            weights = 1.0 / pixel_uncertainty
            fit_design = design * weights[:, None]
            fit_y = y * weights

        parameters, _, rank, _ = np.linalg.lstsq(fit_design, fit_y, rcond=None)
        if rank < 2:
            raise ValueError("sky-line positions do not constrain an affine refinement")
        shift, stretch = (float(value) for value in parameters)
        residuals = y - design @ parameters

        normal = fit_design.T @ fit_design
        if pixel_uncertainty is not None:
            covariance = np.linalg.inv(normal)
        elif observed_pixels.size > 2:
            variance = np.sum(residuals**2) / (observed_pixels.size - 2)
            covariance = np.linalg.inv(normal) * variance
        else:
            covariance = None

    return SkyRefinement(
        mode=mode,
        reference_pixel=reference_pixel,
        shift=shift,
        stretch=stretch,
        rms_pixel=float(np.sqrt(np.mean(residuals**2))),
        n_lines=observed_pixels.size,
        covariance=covariance,
        observed_pixels=observed_pixels,
        master_pixels=master_pixels,
        wavelengths=wavelengths,
    )
