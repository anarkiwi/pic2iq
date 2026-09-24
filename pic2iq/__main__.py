"""Command line interface: pic2iq [image.png | --text TEXT] output_base."""

import argparse
import sys

from pic2iq.core import (
    ORIENTATIONS,
    image_to_iq,
    load_image,
    prepare_image,
    render_text,
    write_sigmf,
)


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="pic2iq",
        description="Convert an image to a SigMF I/Q recording whose "
        "spectrogram shows the image.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("image", nargs="?", help="input image (PNG etc.)")
    src.add_argument("--text", help="render this text instead of an image")
    parser.add_argument("output", help="output base path (no .sigmf-* suffix)")
    parser.add_argument(
        "--fft-size",
        type=int,
        default=512,
        help="spectrogram FFT size; image is scaled to this many rows",
    )
    parser.add_argument(
        "--samples-per-column",
        type=int,
        default=None,
        help="samples per image column (default: FFT size)",
    )
    parser.add_argument("--sample-rate", type=float, default=1e6)
    parser.add_argument("--center-freq", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-range", type=float, default=60.0, help="dB, black to white"
    )
    parser.add_argument(
        "--level", type=float, default=0.0, help="white pixel tone level, dBFS"
    )
    parser.add_argument(
        "--orientation",
        choices=ORIENTATIONS,
        default="inspectrum",
        help="inspectrum: time left to right; waterfall: time top to bottom",
    )
    parser.add_argument(
        "--iterations", type=int, default=32, help="Griffin-Lim iterations"
    )
    parser.add_argument(
        "--font-size", type=int, default=None, help="text height in FFT bins"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point."""
    args = parse_args(argv)
    if args.text is not None:
        img = render_text(
            args.text, args.font_size or args.fft_size // 4, args.fft_size
        )
        pixels = prepare_image(img, args.fft_size, args.orientation)
    else:
        pixels = load_image(args.image, args.fft_size, args.orientation)
    samples = image_to_iq(
        pixels,
        samples_per_column=args.samples_per_column,
        dynamic_range=args.dynamic_range,
        level=args.level,
        iterations=args.iterations,
        seed=args.seed,
    )
    write_sigmf(
        args.output,
        samples,
        args.sample_rate,
        args.center_freq,
        description=args.text or args.image,
    )
    print(f"{args.output}.sigmf-meta: {len(samples)} samples", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
