# Method

Each image row is a frequency bin of an `fft_size`-point Hann-windowed STFT
(top row = +fs/2) and each column spans `samples_per_column` samples. Pixel
intensity `p` is mapped to a target magnitude on a dB scale,
`level - dynamic_range * (1 - p)` dB, with black pixels silent.

Phases are initialised as phase-continuous tones centred on each bin with
random start phases, then refined with fast Griffin-Lim (Perraudin, Balazs &
Søndergaard, 2013) using the same window and FFT size that inspectrum uses
(`scipy.signal.ShortTimeFFT`, hop `fft_size / 4`). Without refinement, the
Hann-window leakage between adjacent tones with random relative phase shows up
as horizontal banding. Griffin-Lim picks phases that keep the magnitude
consistent across rows.

inspectrum shows `10 log10 |X / N|^2`, so a white pixel with `--level 0`
appears at about -6 dB, which falls inside its default -100..0 dB colour range.

## Headless inspectrum screenshots

`tools/inspectrum_screenshot.sh <base> <out.png> [WxH]` builds
`docker/inspectrum` (Ubuntu inspectrum, Xvfb) and captures the window with
the recording loaded at default settings:

```sh
pic2iq --text "hello nate" --font-size 192 out/hello_nate
tools/inspectrum_screenshot.sh out/hello_nate docs/hello_nate_inspectrum.png
```
