"""Physics and interface tests for pic2iq."""

import numpy as np
import pytest
from PIL import Image
import sigmf
from sigmf.sigmffile import fromfile

from pic2iq.__main__ import main
from pic2iq.core import (
    image_to_iq,
    load_image,
    pixel_amplitudes,
    prepare_image,
    render_text,
    spectrogram,
    write_sigmf,
)

FFT = 64
COLS = 12


def tone_image(row, fft_size=FFT, cols=COLS):
    """Image with a single fully lit row."""
    pixels = np.zeros((fft_size, cols), dtype=np.float32)
    pixels[row] = 1.0
    return pixels


def peak_freq(samples):
    """Frequency (cycles/sample) of the strongest spectral component."""
    spec = np.abs(np.fft.fft(samples))
    return np.fft.fftfreq(len(samples))[np.argmax(spec)]


@pytest.mark.parametrize("row", [4, 20, 44, 60])
def test_row_maps_to_bin_frequency(row):
    """Row 0 is the highest frequency; rows below centre are negative."""
    samples = image_to_iq(tone_image(row), iterations=4)
    expected = (FFT // 2 - 1 - row) / FFT
    assert peak_freq(samples) == pytest.approx(expected, abs=0.5 / FFT)
    assert np.argmax(spectrogram(samples, FFT)[:, COLS // 2]) == row


@pytest.mark.parametrize("row", [8, 56])
def test_single_row_is_hann_least_squares_fit(row):
    """A lone lit row settles at the least-squares tone under Hann leakage.

    A tone's Hann STFT also fills its neighbouring bins, so the target
    (one lit bin) is unreachable; Griffin-Lim converges to the amplitude
    minimising the error against the tone's bin-centre leakage profile.
    """
    leak = np.abs(np.fft.fft(np.hanning(FFT + 1)[:-1]))[[0, 1, -1]]
    leak /= leak[0]
    samples = image_to_iq(tone_image(row), level=0.0, iterations=8)
    mags = spectrogram(samples, FFT)[row, 1:COLS]
    np.testing.assert_allclose(mags, 1.0 / np.sum(leak**2), rtol=0.01)


def test_wide_band_reaches_target_level():
    """Rows inside a wide lit band read the full-scale target magnitude."""
    pixels = np.zeros((FFT, COLS), dtype=np.float32)
    pixels[16:48] = 1.0
    spec = spectrogram(image_to_iq(pixels, level=-6.0), FFT)[20:44, 1:COLS]
    assert np.median(20 * np.log10(spec)) == pytest.approx(-6.0, abs=1.0)


def test_tone_is_steady_and_isolated():
    """A lit row is a constant-envelope tone confined to its bin."""
    row = 16
    samples = image_to_iq(tone_image(row), iterations=8)
    spec = spectrogram(samples, FFT)[:, 1:COLS]
    assert np.ptp(spec[row]) < 1e-3 * spec[row].mean()
    far = np.delete(spec, np.arange(row - 2, row + 3), axis=0)
    assert far.max() < 1e-3 * spec[row].min()


@pytest.fixture(name="text_pixels", scope="module")
def fixture_text_pixels():
    """Rendered text scaled to the test FFT size."""
    return prepare_image(render_text("Hi", 16), FFT)


def spectrogram_db(pixels, **kwargs):
    """Synthesise ``pixels`` and return its spectrogram in dB."""
    samples = image_to_iq(pixels, **kwargs)
    mag = spectrogram(samples, FFT)[:, : pixels.shape[1]]
    return 20 * np.log10(np.maximum(mag, 1e-15))


def test_text_round_trip(text_pixels):
    """Lit text pixels hit their target level; dark pixels stay dark."""
    dynamic_range, level = 40.0, -6.0
    db = spectrogram_db(
        text_pixels, dynamic_range=dynamic_range, level=level, iterations=16
    )
    target = level - dynamic_range * (1 - text_pixels)
    lit = text_pixels > 0.9
    dark = text_pixels == 0
    assert lit.sum() > 50 and dark.sum() > 50
    assert abs(np.mean(db[lit] - target[lit])) < 2.0
    assert np.median(db[dark]) < level - dynamic_range


def test_griffin_lim_improves_consistency(text_pixels):
    """Griffin-Lim flattens the lit-pixel level spread."""
    lit = text_pixels > 0.9
    std0 = np.std(spectrogram_db(text_pixels, iterations=0)[lit])
    std16 = np.std(spectrogram_db(text_pixels, iterations=16)[lit])
    assert std16 < 0.5 * std0


@pytest.mark.parametrize("spc", [16, 64, 100])
def test_samples_per_column_sets_length(spc):
    """Output length is width times samples per column."""
    samples = image_to_iq(tone_image(10), samples_per_column=spc, iterations=1)
    assert samples.dtype == np.complex64
    assert len(samples) == COLS * spc


def test_default_samples_per_column_is_fft_size():
    """Samples per column defaults to the FFT size."""
    assert len(image_to_iq(tone_image(10), iterations=0)) == COLS * FFT


@pytest.mark.parametrize("height", [40, 64, 101])
def test_render_text_height_centres_text(height):
    """Canvas is exactly ``height`` rows with the text centred."""
    img = render_text("Hi", 16, height)
    arr = np.asarray(img)
    assert img.mode == "L" and img.height == height
    rows = np.flatnonzero(arr.max(axis=1) > 128)
    assert abs((rows[0] + rows[-1]) / 2 - (height - 1) / 2) <= 1.0
    cols = np.flatnonzero(arr.max(axis=0) > 0)
    assert cols[0] > 0 and cols[-1] < img.width - 1


def test_render_text_default_height_pads_by_size():
    """Default canvas pads the text height by the font size."""
    size = 20
    arr = np.asarray(render_text("Hi", size))
    rows = np.flatnonzero(arr.max(axis=1) > 0)
    assert arr.shape[0] - (rows[-1] - rows[0] + 1) == pytest.approx(size, abs=2)


def test_pixel_amplitudes():
    """Black is silent, white is ``level``, mid-grey is halfway in dB."""
    level, dynamic_range = -3.0, 50.0
    amps = pixel_amplitudes(np.array([0.0, 1.0, 0.5]), dynamic_range, level)
    assert amps[0] == 0.0
    assert amps[1] == pytest.approx(10 ** (level / 20))
    assert 20 * np.log10(amps[2]) == pytest.approx(level - dynamic_range / 2)


def asymmetric_image():
    """Time down, frequency right: a lit pixel at earliest time, lowest freq."""
    arr = np.zeros((24, FFT), dtype=np.uint8)
    arr[0, 0] = 255
    return Image.fromarray(arr, mode="L")


def test_prepare_image_inspectrum_shape_and_scale():
    """Image is scaled to fft_size rows keeping aspect."""
    img = Image.new("L", (30, 16), 255)
    pixels = prepare_image(img, FFT)
    assert pixels.shape == (FFT, 120)
    assert pixels.dtype == np.float32
    np.testing.assert_allclose(pixels, 1.0, atol=1e-6)


def test_prepare_image_waterfall_rotates():
    """Waterfall time-down/frequency-right becomes time-left/high-top."""
    pixels = prepare_image(asymmetric_image(), FFT, "waterfall")
    assert pixels.shape == (FFT, 24)
    peak = np.unravel_index(np.argmax(pixels), pixels.shape)
    assert peak == (FFT - 1, 0)
    inspect = prepare_image(asymmetric_image(), FFT, "inspectrum")
    assert np.unravel_index(np.argmax(inspect), inspect.shape) == (0, 0)


def test_prepare_image_bad_orientation():
    """Unknown orientations are rejected."""
    with pytest.raises(ValueError):
        prepare_image(asymmetric_image(), FFT, "sideways")


def test_load_image_alpha_composites_to_black(tmp_path):
    """Transparent pixels become black."""
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[:, :4, 3] = 255
    path = tmp_path / "alpha.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    pixels = load_image(path, 8)
    np.testing.assert_allclose(pixels[:, :4], 1.0, atol=1e-6)
    np.testing.assert_allclose(pixels[:, 4:], 0.0, atol=1e-6)


def test_load_image_greyscale(tmp_path):
    """Greyscale images load unchanged at native size."""
    arr = np.tile(np.array([0, 128, 255], dtype=np.uint8), (6, 2))
    path = tmp_path / "grey.png"
    Image.fromarray(arr, mode="L").save(path)
    pixels = load_image(path, 6)
    np.testing.assert_allclose(pixels, arr / 255.0, atol=1e-6)


def test_write_sigmf_round_trip(tmp_path):
    """SigMF samples and metadata survive a write/read."""
    rng = np.random.default_rng(1)
    samples = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(
        np.complex64
    )
    base = tmp_path / "rec"
    write_sigmf(base, samples, 2e6, 433.92e6, description="hello")
    meta = fromfile(f"{base}.sigmf-meta")
    np.testing.assert_array_equal(meta.read_samples(), samples)
    assert meta.get_global_field(sigmf.SAMPLE_RATE_KEY) == 2e6
    assert meta.get_global_field(sigmf.DATATYPE_KEY) == "cf32_le"
    assert meta.get_global_field(sigmf.DESCRIPTION_KEY) == "hello"
    assert meta.get_captures()[0][sigmf.FREQUENCY_KEY] == 433.92e6


def test_write_sigmf_overwrites(tmp_path):
    """Writing to an existing base replaces it."""
    base = tmp_path / "again"
    write_sigmf(base, np.ones(16, dtype=np.complex64), 1e6, description="first")
    second = np.full(8, 2j, dtype=np.complex64)
    write_sigmf(base, second, 2e6, description="second")
    meta = fromfile(f"{base}.sigmf-meta")
    np.testing.assert_array_equal(meta.read_samples(), second)
    assert meta.get_global_field(sigmf.SAMPLE_RATE_KEY) == 2e6
    assert meta.get_global_field(sigmf.DESCRIPTION_KEY) == "second"


def test_write_sigmf_without_description(tmp_path):
    """Description is omitted when not given."""
    base = tmp_path / "bare"
    write_sigmf(base, np.zeros(8, dtype=np.complex64), 1e6)
    meta = fromfile(f"{base}.sigmf-meta")
    assert meta.get_global_field(sigmf.DESCRIPTION_KEY) is None


CLI_OPTS = ["--fft-size", "32", "--iterations", "2", "--sample-rate", "48000"]


def test_cli_image(tmp_path):
    """CLI converts an image file to a SigMF recording."""
    path = tmp_path / "in.png"
    asymmetric_image().save(path)
    base = tmp_path / "img"
    argv = [str(path), str(base), *CLI_OPTS, "--orientation", "waterfall"]
    assert main(argv) == 0
    meta = fromfile(f"{base}.sigmf-meta")
    assert meta.get_global_field(sigmf.SAMPLE_RATE_KEY) == 48000
    assert meta.get_global_field(sigmf.DESCRIPTION_KEY) == str(path)
    assert len(meta.read_samples()) % 32 == 0


def test_cli_text(tmp_path):
    """CLI renders text, and re-running overwrites the output."""
    base = tmp_path / "txt"
    argv = ["--text", "ok", str(base), *CLI_OPTS, "--samples-per-column", "8"]
    assert main(argv) == 0
    assert (tmp_path / "txt.sigmf-data").stat().st_size > 0
    meta = fromfile(f"{base}.sigmf-meta")
    assert meta.get_global_field(sigmf.DESCRIPTION_KEY) == "ok"
    assert len(meta.read_samples()) % 8 == 0
    assert main(argv) == 0


def test_cli_requires_source(tmp_path):
    """CLI exits with a usage error without image or text."""
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "out")])
    assert exc.value.code == 2
