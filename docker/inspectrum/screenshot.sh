#!/bin/sh
# usage: screenshot.sh <recording.sigmf-meta> <out.png> [WIDTHxHEIGHT]
set -e
GEOM="${3:-1600x900}"
Xvfb :99 -screen 0 "${GEOM}x24" -nolisten tcp &
export DISPLAY=:99
sleep 1
inspectrum "$1" &
PID=$!
WID=""
while [ -z "$WID" ]; do sleep 0.5; WID=$(xdotool search --pid "$PID" --onlyvisible 2>/dev/null | tail -1); done
xdotool windowmove "$WID" 0 0 windowsize "$WID" "${GEOM%x*}" "${GEOM#*x}"
sleep 3
import -window root "$2"
kill "$PID"
