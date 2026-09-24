#!/bin/sh
# usage: tools/inspectrum_screenshot.sh <base> <out.png> [WIDTHxHEIGHT]
# Renders <base>.sigmf-meta in inspectrum (headless, in Docker) to a PNG.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
docker build -q -t pic2iq-inspectrum "$HERE/docker/inspectrum" >/dev/null
DIR=$(cd "$(dirname "$1")" && pwd)
OUT=$(cd "$(dirname "$2")" && pwd)
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$DIR:/in:ro" -v "$OUT:/out" pic2iq-inspectrum \
    "/in/$(basename "$1").sigmf-meta" "/out/$(basename "$2")" "${3:-1440x560}" 2>/dev/null
