"""Image to I/Q synthesis via inverse short-time Fourier transform."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
import sigmf
from sigmf import SigMFFile

ORIENTATIONS = ("inspectrum", "waterfall")


def load_image(path, fft_size, orientation="inspectrum"):
    """Load an image as a (fft_size, width) float array in [0, 1].

    Rows are frequency bins (top row highest), columns are time. Alpha is
    composited onto black. For ``waterfall`` orientation the input is read
    with time running down and frequency running right.
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, "black"), img)
    return prepare_image(img.convert("L"), fft_size, orientation)


def prepare_image(img, fft_size, orientation="inspectrum"):
    """Orient and resize a greyscale PIL image to ``fft_size`` rows."""
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}")
    if orientation == "waterfall":
        img = img.transpose(Image.Transpose.ROTATE_90)
    width = max(1, round(img.width * fft_size / img.height))
    img = img.resize((width, fft_size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def render_text(text, size=64, height=None):
    """Render white text on black as a greyscale PIL image.

    The text is centred vertically in ``height`` rows (default: text height
    plus ``size`` rows of padding) with ``size // 2`` columns either side.
    """
    font = ImageFont.load_default(size=size)
    left, top, right, bottom = font.getbbox(text)
    height = bottom - top + size if height is None else height
    pad = size // 2
    img = Image.new("L", (right - left + 2 * pad, height), 0)
    xy = (pad - left, (height - (bottom - top)) // 2 - top)
    ImageDraw.Draw(img).text(xy, text, fill=255, font=font)
    return img


def pixel_amplitudes(pixels, dynamic_range=60.0, level=0.0):
    """Map [0, 1] pixel intensity to linear tone amplitude on a dB scale."""
    db = level - dynamic_range * (1.0 - pixels)
    return np.where(pixels > 0, 10.0 ** (db / 20.0), 0.0)


def make_stft(fft_size, hop):
    """Hann-windowed, centred, magnitude-scaled STFT as used by inspectrum."""
    return ShortTimeFFT(
        hann(fft_size, sym=False),
        hop,
        fs=1.0,
        fft_mode="centered",
        scale_to="magnitude",
    )


def spectrogram(samples, fft_size, stride=None):
    """|STFT| with row 0 the highest frequency, one column per ``stride``."""
    stft = make_stft(fft_size, fft_size if stride is None else stride)
    return np.abs(stft.stft(samples))[::-1]


def image_to_iq(
    pixels,
    samples_per_column=None,
    dynamic_range=60.0,
    level=0.0,
    overlap=4,
    iterations=32,
    momentum=0.99,
    seed=0,
):
    """Synthesise complex baseband samples whose spectrogram is ``pixels``.

    ``pixels`` has shape (fft_size, width): row 0 is the highest frequency
    bin, column 0 the earliest time. Pixels set target Hann STFT magnitudes
    (full scale is ``10**(level/20)``), fitted in the least-squares sense
    where the target is not a consistent STFT. Phases start as phase-continuous tones on
    each bin centre and are refined by fast Griffin-Lim (Perraudin et al.
    2013) so a Hann STFT of the same size reproduces the image magnitudes.
    """
    fft_size, width = pixels.shape
    spc = fft_size if samples_per_column is None else samples_per_column
    hop = max(1, fft_size // overlap)
    stft = make_stft(fft_size, hop)
    n = width * spc
    frames = np.arange(stft.p_min, stft.p_max(n))
    cols = np.clip((frames * hop) // spc, 0, width - 1)
    mag = pixel_amplitudes(pixels[::-1], dynamic_range, level)[:, cols]
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, (fft_size, 1))
    phase = phase + 2 * np.pi * stft.f[:, None] * (frames * hop - stft.m_num_mid)
    angles = np.exp(1j * phase)
    prev = 0.0
    for _ in range(iterations):
        rebuilt = stft.stft(stft.istft(mag * angles, k1=n))
        angles = rebuilt - (momentum / (1 + momentum)) * prev
        angles /= np.maximum(np.abs(angles), 1e-12)
        prev = rebuilt
    return stft.istft(mag * angles, k1=n).astype(np.complex64)


def write_sigmf(base, samples, sample_rate, center_freq=0.0, description=None):
    """Write ``samples`` as ``base.sigmf-data`` / ``base.sigmf-meta``."""
    data_path = f"{base}.sigmf-data"
    samples.astype(np.complex64).tofile(data_path)
    global_info = {
        sigmf.DATATYPE_KEY: "cf32_le",
        sigmf.SAMPLE_RATE_KEY: float(sample_rate),
        sigmf.GENERATOR_KEY: "pic2iq",
    }
    if description:
        global_info[sigmf.DESCRIPTION_KEY] = description
    meta = SigMFFile(data_file=data_path, global_info=global_info)
    meta.add_capture(0, metadata={sigmf.FREQUENCY_KEY: float(center_freq)})
    meta.tofile(f"{base}.sigmf-meta", overwrite=True)
    return meta
