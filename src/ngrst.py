"""Command line for the NGRST discovery loop."""

from pathlib import Path
import argparse

import roman_toolkit


def discover(args):
    if not args.mast:
        print("MAST query skipped (no --mast).")

    reference = args.reference
    if args.mast:
        if args.ra is None or args.dec is None:
            raise SystemExit("--mast requires --ra and --dec.")
        downloaded = roman_toolkit.fetch_mast_exposure(
            filter_name=args.filter,
            ra=args.ra,
            dec=args.dec,
            radius_deg=args.radius,
            download_dir=roman_toolkit.RAW_DIRECTORY,
        )
        if downloaded is None:
            print("No MAST file downloaded. Nothing to discover.")
            return 0
        exposure = downloaded
    else:
        if args.input is None:
            raise SystemExit("An exposure path is required unless --mast is set.")
        exposure = Path(args.input)

    roman_toolkit.run_pipeline(
        exposure,
        top=args.top,
        reference=reference,
        online=args.crossmatch_online,
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ngrst",
        description="NGRST discovery loop",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    discover_parser = commands.add_parser(
        "discover",
        help="Turn one exposure into a sky catalog",
    )
    discover_parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Local .asdf, .fits, or .npz exposure",
    )
    discover_parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of candidate cutouts to write",
    )
    discover_parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Offline reference catalog (.ecsv or .parquet). "
            "Defaults to tests/fixtures/reference_catalog.ecsv when present."
        ),
    )
    discover_parser.add_argument(
        "--mast",
        action="store_true",
        help="Query MAST for one Roman exposure, then run the local path",
    )
    discover_parser.add_argument(
        "--crossmatch-online",
        action="store_true",
        help="Cross-match sky positions against SIMBAD",
    )
    discover_parser.add_argument(
        "--filter",
        default="F158",
        help="Roman filter used by --mast",
    )
    discover_parser.add_argument("--ra", type=float, help="Cone center RA in degrees")
    discover_parser.add_argument("--dec", type=float, help="Cone center Dec in degrees")
    discover_parser.add_argument(
        "--radius",
        type=float,
        default=0.05,
        help="MAST cone radius in degrees",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "discover":
        return discover(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
