#!/usr/bin/env python3
"""
make8d.py - turn a music file into animated binaural "8D audio".

Convolves the track with real head-related impulse responses (HRIRs) from a
.sofa file, sweeping the source continuously around the listener's head.
Bass below the crossover stays mono and centered.

Usage:
    ./make8d.py song.wav -s kemar.sofa
    ./make8d.py song.mp3 -s kemar.sofa --period 14 --out spun.wav
    ./make8d.py song.wav -s kemar.sofa --preview 20     # first 20s only
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import soundfile as sf


# ---------------------------------------------------------------- audio input

def read_audio(path):
    """Read any audio file. libsndfile handles wav/flac/aiff/ogg/mp3; anything
    else (m4a, aac, wma, opus) gets decoded through ffmpeg first."""
    try:
        audio, fs = sf.read(path, always_2d=True, dtype="float64")
        return audio, float(fs)
    except Exception:
        pass

    if not shutil.which("ffmpeg"):
        raise SystemExit(
            f"Cannot read {path}. libsndfile does not know this format and "
            f"ffmpeg is not installed. Try: brew install ffmpeg"
        )

    print("  format not native to libsndfile, decoding via ffmpeg", file=sys.stderr)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", path,
             "-c:a", "pcm_f32le", tmp.name],
            check=True,
        )
        audio, fs = sf.read(tmp.name, always_2d=True, dtype="float64")
        return audio, float(fs)
    except subprocess.CalledProcessError:
        raise SystemExit(f"ffmpeg could not decode {path}.")
    finally:
        os.unlink(tmp.name)


# ----------------------------------------------------------------- SOFA input

def load_hrirs(path):
    """Return (irs, fs, azimuths) for the horizontal plane, sorted by azimuth.

    irs: (M, 2, N) float64, M measurements of a stereo (L,R) impulse response
    azimuths: (M,) degrees, 0 = front, increasing counterclockwise
    """
    with h5py.File(path, "r") as f:
        if "Data.IR" not in f:
            raise SystemExit(
                f"{path} has no Data.IR dataset. That is not a SOFA HRIR file."
            )
        ir = np.asarray(f["Data.IR"], dtype=np.float64)        # (M, R, N)
        fs = float(np.asarray(f["Data.SamplingRate"]).flatten()[0])
        pos = np.asarray(f["SourcePosition"], dtype=np.float64)  # (M, 3)

    if ir.ndim != 3 or ir.shape[1] < 2:
        raise SystemExit(f"Expected (M,2,N) receiver data, got {ir.shape}.")
    ir = ir[:, :2, :]

    az, el = pos[:, 0], pos[:, 1]

    # Keep the horizontal plane. Widen the tolerance if a sparse set misses 0.
    for tol in (2.0, 5.0, 10.0, 20.0):
        keep = np.abs(el) <= tol
        if keep.sum() >= 8:
            break
    else:
        raise SystemExit("Could not find a usable horizontal plane in this SOFA file.")

    ir, az = ir[keep], np.mod(az[keep], 360.0)

    # Collapse duplicate azimuths, then sort.
    az, idx = np.unique(np.round(az, 3), return_index=True)
    return ir[idx], fs, az


def resample_last_axis(x, fs_in, fs_out):
    """Frequency-domain resample along the last axis. Fine for short HRIRs."""
    if abs(fs_in - fs_out) < 1e-6:
        return x
    n_in = x.shape[-1]
    n_out = int(round(n_in * fs_out / fs_in))
    spec = np.fft.rfft(x, axis=-1)
    out_bins = n_out // 2 + 1
    resized = np.zeros(x.shape[:-1] + (out_bins,), dtype=complex)
    k = min(spec.shape[-1], out_bins)
    resized[..., :k] = spec[..., :k]
    return np.fft.irfft(resized, n=n_out, axis=-1) * (n_out / n_in)


# ------------------------------------------------------------ band splitting

def lowpass_fir(cutoff_hz, fs, taps=1025):
    """Linear-phase windowed-sinc lowpass. Odd length so delay is exactly
    (taps-1)/2, which we cancel with a centered convolution."""
    n = np.arange(taps) - (taps - 1) / 2
    fc = cutoff_hz / fs
    h = 2 * fc * np.sinc(2 * fc * n)
    h *= np.blackman(taps)
    return h / h.sum()


def split_bands(audio, cutoff_hz, fs):
    """Return (low_mono, high_stereo).

    high = audio - lowpass(audio) per channel, so low + high reconstructs the
    input exactly. No crossover phase error, unlike cascaded IIR filters.
    """
    h = lowpass_fir(cutoff_hz, fs)
    pad = (len(h) - 1) // 2

    def lp(x):
        padded = np.pad(x, pad, mode="reflect")
        return np.convolve(padded, h, mode="valid")

    low_per_ch = np.stack([lp(audio[:, c]) for c in range(audio.shape[1])], axis=1)
    high = audio - low_per_ch
    low_mono = low_per_ch.mean(axis=1)
    return low_mono, high


# -------------------------------------------------------------- the orbit

def orbit(high, fs, irs, azimuths, period_s, block=4096, progress=True):
    """Weighted overlap-add convolution with a time-varying HRIR.

    Periodic Hann at 50% overlap sums to exactly 1.0, so the filter can change
    every block without clicks or level ripple.
    """
    hop = block // 2
    n_ir = irs.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(block + n_ir - 1)))

    # Pre-transform every HRIR once.
    ir_spec = np.fft.rfft(irs, n=nfft, axis=-1)           # (M, 2, nfft/2+1)

    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(block) / block)

    n = high.shape[0]
    n_blocks = int(np.ceil(n / hop)) + 1
    padded = np.pad(high, ((0, n_blocks * hop + block), (0, 0)))
    out = np.zeros((padded.shape[0] + nfft, 2))

    az_ext = np.concatenate([azimuths, [azimuths[0] + 360.0]])

    for b in range(n_blocks):
        start = b * hop
        chunk = padded[start:start + block]
        if not chunk.any():
            continue

        # Angle at this block's midpoint.
        t = (start + block / 2) / fs
        az = (360.0 * t / period_s) % 360.0

        # Bracketing measured angles, linearly blended.
        j = np.searchsorted(az_ext, az, side="right") - 1
        j = min(max(j, 0), len(azimuths) - 1)
        lo_az, hi_az = az_ext[j], az_ext[j + 1]
        frac = 0.0 if hi_az == lo_az else (az - lo_az) / (hi_az - lo_az)
        spec = (1 - frac) * ir_spec[j] + frac * ir_spec[(j + 1) % len(azimuths)]

        # Feed the panner a mono sum: a stereo source has its own width baked
        # in, which fights the HRTF cue we are adding.
        mono = chunk.mean(axis=1) * win
        mono_spec = np.fft.rfft(mono, n=nfft)

        wet = np.fft.irfft(mono_spec[None, :] * spec, n=nfft, axis=-1)  # (2, nfft)
        out[start:start + nfft] += wet.T

        if progress and b % 200 == 0:
            pct = 100.0 * b / n_blocks
            print(f"\r  orbiting... {pct:5.1f}%", end="", file=sys.stderr, flush=True)

    if progress:
        print("\r  orbiting... 100.0%", file=sys.stderr)
    return out[:n]


# ---------------------------------------------------------------- reverb

def reverb(x, fs, amount):
    """Cheap decorrelated multi-tap reverb. Enough to put the orbit in a room."""
    if amount <= 0:
        return x
    taps_ms = [23, 41, 67, 97, 131, 173]
    gains = [0.50, 0.38, 0.28, 0.20, 0.14, 0.09]
    wet = np.zeros_like(x)
    for i, (ms, g) in enumerate(zip(taps_ms, gains)):
        d = int(fs * ms / 1000.0)
        # Offset the right channel slightly so the tail is not a mono blob.
        dr = d + int(fs * (3 + i) / 1000.0)
        wet[d:, 0] += g * x[:len(x) - d, 0]
        wet[dr:, 1] += g * x[:len(x) - dr, 1]
    return x + amount * wet


# ------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(
        description="Turn a track into animated binaural 8D audio.")
    p.add_argument("input", help="input audio file (wav/flac/aiff/ogg/mp3/m4a/...)")
    p.add_argument("-s", "--sofa", required=True, help="path to a .sofa HRIR file")
    p.add_argument("-o", "--out", help="output path (default: <input>_8d.wav)")
    p.add_argument("--period", type=float, default=10.0,
                   help="seconds per full rotation (default: 10)")
    p.add_argument("--bass", type=float, default=120.0,
                   help="Hz below which audio stays mono and centered (default: 120)")
    p.add_argument("--reverb", type=float, default=0.25,
                   help="reverb amount 0..1, 0 disables (default: 0.25)")
    p.add_argument("--preview", type=float, default=0.0,
                   help="only process the first N seconds")
    args = p.parse_args()

    out_path = args.out or (args.input.rsplit(".", 1)[0] + "_8d.wav")

    print(f"Reading {args.input}", file=sys.stderr)
    audio, fs = read_audio(args.input)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    audio = audio[:, :2]
    if args.preview > 0:
        audio = audio[:int(args.preview * fs)]
    print(f"  {audio.shape[0]/fs:.1f}s @ {fs:.0f} Hz", file=sys.stderr)

    print(f"Reading {args.sofa}", file=sys.stderr)
    irs, ir_fs, az = load_hrirs(args.sofa)
    print(f"  {len(az)} horizontal angles @ {ir_fs:.0f} Hz, "
          f"{irs.shape[-1]} taps", file=sys.stderr)
    if abs(ir_fs - fs) > 1e-6:
        print(f"  resampling HRIRs {ir_fs:.0f} -> {fs:.0f} Hz", file=sys.stderr)
        irs = resample_last_axis(irs, ir_fs, fs)

    print(f"Splitting at {args.bass:.0f} Hz", file=sys.stderr)
    low_mono, high = split_bands(audio, args.bass, fs)

    print(f"Rotating every {args.period:.1f}s", file=sys.stderr)
    wet = orbit(high, fs, irs, az, args.period)
    wet = reverb(wet, fs, args.reverb)

    # Put the untouched bass back, centered.
    wet[:, 0] += low_mono
    wet[:, 1] += low_mono

    peak = np.abs(wet).max()
    if peak > 0.99:
        print(f"  normalizing (peak was {peak:.2f})", file=sys.stderr)
        wet *= 0.99 / peak

    sf.write(out_path, wet, int(fs), subtype="PCM_24")
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
