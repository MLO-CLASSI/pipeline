"""Command-line interface for the spectrograph pipeline."""

import argparse
from pathlib import Path

from .extraction import optimal_extract_fibers
from .geometry import evenly_spaced_centers, infer_half_width, parse_centers
from .io import read_image, write_extracted_fits
from .variance import estimate_variance_adu


def _extension(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectrograph-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract",
        help="Horne optimal extraction of flat fiber traces",
    )
    extract.add_argument("input", type=Path, help="Input 2-D FITS image")
    extract.add_argument("output", type=Path, help="Output extracted FITS file")
    extract.add_argument(
        "--data-ext",
        type=_extension,
        default=0,
        help="Science-image FITS extension name or index (default: 0)",
    )
    extract.add_argument(
        "--variance-ext",
        type=_extension,
        help="Variance-plane FITS extension name or index",
    )
    extract.add_argument(
        "--mask-ext",
        type=_extension,
        help="Boolean/bad-pixel mask FITS extension name or index",
    )

    geometry = extract.add_argument_group("trace geometry")
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
        help="Fiber center-to-center spacing in detector pixels",
    )
    geometry.add_argument(
        "--n-fibers",
        type=int,
        default=7,
        help="Number of evenly spaced fibers (default: 7)",
    )
    geometry.add_argument(
        "--half-width",
        type=int,
        help=(
            "Cross-dispersion half-width of each local extraction cutout. "
            "By default, infer it from half the minimum fiber spacing."
        ),
    )

    noise = extract.add_argument_group("variance fallback")
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
    extract.add_argument("--overwrite", action="store_true")
    return parser


def _get_centers(args: argparse.Namespace):
    if args.centers is not None:
        if args.center is not None or args.spacing is not None:
            raise ValueError("use either --centers or --center/--spacing, not both")
        return parse_centers(args.centers)

    if args.center is None or args.spacing is None:
        raise ValueError("provide --centers or both --center and --spacing")

    return evenly_spaced_centers(args.center, args.spacing, args.n_fibers)


def _run_extract(args: argparse.Namespace) -> None:
    centers = _get_centers(args)
    data, variance, mask, header = read_image(
        args.input,
        data_ext=args.data_ext,
        variance_ext=args.variance_ext,
        mask_ext=args.mask_ext,
    )

    if variance is None:
        if args.gain is None or args.read_noise is None:
            raise ValueError(
                "optimal extraction requires a variance estimate: provide "
                "--variance-ext or both --gain and --read-noise"
            )
        variance = estimate_variance_adu(data, args.gain, args.read_noise)

    half_width = args.half_width if args.half_width is not None else infer_half_width(centers)
    spectra = optimal_extract_fibers(
        data,
        variance,
        centers,
        half_width=half_width,
        mask=mask,
    )
    write_extracted_fits(
        args.output,
        spectra,
        centers,
        source_header=header,
        source_path=args.input,
        overwrite=args.overwrite,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "extract":
            _run_extract(args)
        else:
            parser.error(f"unknown command: {args.command}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
