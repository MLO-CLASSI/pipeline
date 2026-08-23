"""Command-line interface for the spectrograph pipeline."""

import argparse
from pathlib import Path

import numpy as np
from astropy.nddata import CCDData

from l1.io import write_l1_fits
from l1.utils import infer_half_width
from l1 import process_l1
from l2 import process_l2


def _extension(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def _add_geometry_arguments(parser: argparse.ArgumentParser) -> None:
    geometry = parser.add_argument_group("trace geometry")
    geometry.add_argument(
        "--centers",
        help="Comma-separated Y centers in pixels, e.g. 100,125,150,...",
    )
    geometry.add_argument(
        "--center",
        type=float,
        help="Bundle center in Y pixels; use with --spacing",
    )
    geometry.add_argument(
        "--spacing",
        type=float,
        help="Trace center-to-center spacing in detector pixels",
    )
    geometry.add_argument(
        "--n-traces",
        type=int,
        default=7,
        help="Number of evenly spaced traces (default: 7)",
    )
    geometry.add_argument(
        "--half-width",
        type=int,
        help=(
            "Cross-dispersion half-width of each local extraction cutout. "
            "By default, infer it from half the minimum trace spacing."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectrograph-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    l1 = subparsers.add_parser(
        "l1",
        help="L0 FITS -> detector-corrected, optimally extracted L1 spectra",
    )
    l1.add_argument("input", type=Path, help="Input L0 FITS image")
    l1.add_argument("output", type=Path, help="Output L1 FITS product")
    l1.add_argument(
        "--data-ext",
        type=_extension,
        default=0,
        help="Science-image FITS extension name or index (default: 0)",
    )
    l1.add_argument(
        "--variance-ext",
        type=_extension,
        help="Variance-plane FITS extension name or index",
    )
    l1.add_argument(
        "--mask-ext",
        type=_extension,
        help="Boolean/bad-pixel mask FITS extension name or index",
    )
    l1.add_argument(
        "--unit",
        help="Science-image unit if BUNIT is absent or should be overridden",
    )

    detector = l1.add_argument_group("optional L1 detector corrections")
    detector.add_argument("--bias", type=Path, help="Master bias FITS file")
    detector.add_argument("--dark", type=Path, help="Master dark FITS file")
    detector.add_argument("--flat", type=Path, help="Master flat FITS file")
    detector.add_argument(
        "--cal-data-ext",
        type=_extension,
        default=0,
        help="Image extension used for master bias/dark/flat files (default: 0)",
    )
    detector.add_argument(
        "--dark-scale",
        action="store_true",
        help="Scale the master dark by exposure time before subtraction",
    )
    detector.add_argument(
        "--exposure-key",
        default="EXPTIME",
        help="Exposure-time FITS keyword used with --dark-scale (default: EXPTIME)",
    )
    detector.add_argument(
        "--flat-min-value",
        type=float,
        help="Minimum master-flat value accepted by ccdproc.flat_correct",
    )

    _add_geometry_arguments(l1)

    noise = l1.add_argument_group("variance fallback")
    noise.add_argument(
        "--gain",
        type=float,
        help="Detector gain in e-/ADU when no variance extension is supplied",
    )
    noise.add_argument(
        "--read-noise",
        type=float,
        help="Read noise in e- when no variance extension is supplied",
    )
    l1.add_argument("--overwrite", action="store_true")

    l2 = subparsers.add_parser(
        "l2",
        help="L1 spectra -> wavelength/spectrophotometrically calibrated L2 spectra",
    )
    l2.add_argument("input", type=Path, help="Input L1 FITS product")
    l2.add_argument("output", type=Path, help="Output L2 FITS product")
    l2.add_argument("--overwrite", action="store_true")

    return parser


def _get_centers(args: argparse.Namespace):
    if args.centers is not None:
        if args.center is not None or args.spacing is not None:
            raise ValueError("use either --centers or --center/--spacing, not both")
        try:
            centers = np.asarray(map(float, args.centers.split(",")), dtype=float)
        except ValueError as exc:
            raise ValueError("trace centers must be comma-separated numbers") from exc

        if centers.size == 0 or not np.all(np.isfinite(centers)):
            raise ValueError("trace centers must contain finite values")
        return centers

    if args.center is None or args.spacing is None:
        raise ValueError("provide --centers or both --center and --spacing")

    offsets = np.arange(args.n_traces, dtype=float) - (args.n_traces - 1) / 2
    return args.center + args.spacing * offsets


def _read_optional_master(path, data_ext, default_unit):
    if path is None:
        return None
    return CCDData.read(path, hdu=data_ext, unit=default_unit)


def _run_l1(args: argparse.Namespace) -> None:
    centers = _get_centers(args)
    ccd = CCDData.read(args.input, hdu=args.data_ext, unit=args.unit)

    bias = _read_optional_master(args.bias, args.cal_data_ext, ccd.unit)
    dark = _read_optional_master(args.dark, args.cal_data_ext, ccd.unit)
    flat = _read_optional_master(args.flat, args.cal_data_ext, ccd.unit)

    half_width = args.half_width if args.half_width is not None else infer_half_width(centers)
    spectra = process_l1(
        ccd,
        centers,
        half_width=half_width,
        gain=args.gain,
        read_noise=args.read_noise,
        bias=bias,
        dark=dark,
        flat=flat,
        dark_scale=args.dark_scale,
        exposure_key=args.exposure_key,
        flat_min_value=args.flat_min_value,
    )
    write_l1_fits(
        args.output,
        spectra,
        centers,
        source_header=ccd.header,
        source_path=args.input,
        bias_path=args.bias,
        dark_path=args.dark,
        flat_path=args.flat,
        dark_scaled=args.dark_scale,
        overwrite=args.overwrite,
    )


def _run_l2(args: argparse.Namespace) -> None:
    process_l2(args.input, args.output)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "l1":
            _run_l1(args)
        elif args.command == "l2":
            _run_l2(args)
        else:
            parser.error(f"unknown command: {args.command}")
    except (OSError, ValueError, NotImplementedError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
