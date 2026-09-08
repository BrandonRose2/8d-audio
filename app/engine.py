"""
engine.py - 8D audio processing core.

Two paths:
  stereo path     pans a mono sum around the head with KEMAR HRTFs
  ambisonic path  rotates a real recorded sound field (iPhone Spatial Audio)

The ambisonic path is used automatically when the file carries a 4-channel
Apple APAC track, and is strictly better: real room rotation, real height.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
SOFA_PATH = HERE / "kemar.sofa"
DECODER = HERE / "bin" / "decode_track"


# ----------------------------------------------------------------- probing

def probe(path: str) -> dict:
    """Inspect a media file. Reports whether a rotatable sound field exists."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()[:300]}")
    data = json.loads(out.stdout)

    audio, video = [], []
    for s in data.get("streams", []):
        if s.get("codec_type") == "audio":
            audio.append({
                "index": s.get("index"),
                "codec": s.get("codec_name"),
                "channels": s.get("channels"),
                "sample_rate": int(s.get("sample_rate") or 0),
                "bit_rate": int(s.get("bit_rate") or 0),
            })
        elif s.get("codec_type") == "video":
            video.append({"index": s.get("index"),
                          "width": s.get("width"), "height": s.get("height")})

    # Apple Spatial Audio: 4-channel APAC, ambisonic ACN/SN3D.
    ambi_slot = None
    for i, a in enumerate(audio):
        if a["codec"] == "apple_apac" and a["channels"] == 4:
            ambi_slot = i
            break

    return {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "size": int(data.get("format", {}).get("size") or 0),
        "audio": audio,
        "video": video,
        "ambisonic_slot": ambi_slot,
        "has_ambisonic": ambi_slot is not None and DECODER.exists(),
        "has_video": bool(video),
    }


# ------------------------------------------------------------ decoding in

def read_stereo(path: str):
    """Any format. libsndfile first, ffmpeg for everything else."""
    try:
        a, fs = sf.read(path, always_2d=True, dtype="float64")
        return a[:, :2] if a.shape[1] >= 2 else np.repeat(a, 2, axis=1), float(fs)
    except Exception:
        pass
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found and libsndfile cannot read this file")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                        "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", tmp.name],
                       check=True, capture_output=True)
        a, fs = sf.read(tmp.name, always_2d=True, dtype="float64")
        return a[:, :2] if a.shape[1] >= 2 else np.repeat(a, 2, axis=1), float(fs)
    finally:
        os.unlink(tmp.name)


def read_ambisonic(path: str, slot: int):
    """Decode the Apple APAC ambisonic track via CoreAudio (Swift helper)."""
    if not DECODER.exists():
        raise RuntimeError("APAC decoder binary missing")
    tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
    tmp.close()
    try:
        r = subprocess.run([str(DECODER), str(path), tmp.name, str(slot)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"APAC decode failed: {r.stderr.strip()[:200]}")
        fs = 48000.0
        for tok in r.stderr.split():
            if tok.startswith("SR="):
                fs = float(tok[3:])
        a = np.fromfile(tmp.name, dtype=np.float32).reshape(-1, 4).astype(np.float64)
        return a, fs
    finally:
        os.unlink(tmp.name)


# ------------------------------------------------------------ SOFA + utils

def _load_sofa():
    with h5py.File(SOFA_PATH, "r") as f:
        ir = np.asarray(f["Data.IR"], dtype=np.float64)[:, :2, :]
        fs = float(np.asarray(f["Data.SamplingRate"]).flatten()[0])
        pos = np.asarray(f["SourcePosition"], dtype=np.float64)
    return ir, fs, pos


def _resample_last(x, fs_in, fs_out):
    if abs(fs_in - fs_out) < 1e-6:
        return x
    n_in = x.shape[-1]
    n_out = int(round(n_in * fs_out / fs_in))
    spec = np.fft.rfft(x, axis=-1)
    bins = n_out // 2 + 1
    y = np.zeros(x.shape[:-1] + (bins,), dtype=complex)
    k = min(spec.shape[-1], bins)
    y[..., :k] = spec[..., :k]
    return np.fft.irfft(y, n=n_out, axis=-1) * (n_out / n_in)


def _lowpass_fir(fc, fs, taps=1025):
    n = np.arange(taps) - (taps - 1) / 2
    f = fc / fs
    h = 2 * f * np.sinc(2 * f * n) * np.blackman(taps)
    return h / h.sum()


def _split_bands(audio, fc, fs):
    """high = audio - lowpass(audio), so the bands sum back exactly."""
    h = _lowpass_fir(fc, fs)
    pad = (len(h) - 1) // 2
    def lp(x):
        return np.convolve(np.pad(x, pad, mode="reflect"), h, mode="valid")
    low = np.stack([lp(audio[:, c]) for c in range(audio.shape[1])], axis=1)
    return low.mean(axis=1), audio - low


def _reverb(x, fs, amount):
    if amount <= 0:
        return x
    wet = np.zeros_like(x)
    for i, (ms, g) in enumerate(zip([23, 41, 67, 97, 131, 173],
                                    [.50, .38, .28, .20, .14, .09])):
        d = int(fs * ms / 1000)
        dr = d + int(fs * (3 + i) / 1000)
        wet[d:, 0] += g * x[:len(x) - d, 0]
        wet[dr:, 1] += g * x[:len(x) - dr, 1]
    return x + amount * wet


def _fft_conv(x, h, block=1 << 16):
    nfft = 1 << int(np.ceil(np.log2(block + len(h) - 1)))
    H = np.fft.rfft(h, n=nfft)
    out = np.zeros(len(x) + len(h))
    for s in range(0, len(x), block):
        y = np.fft.irfft(np.fft.rfft(x[s:s + block], n=nfft) * H, n=nfft)
        out[s:s + len(y)] += y[:len(out) - s]
    return out[:len(x)]


# --------------------------------------------------------- stereo pipeline

def process_stereo(audio, fs, period=10.0, bass=120.0, reverb=0.25, progress=None):
    def say(p, m):
        if progress:
            progress(p, m)

    say(0.10, "Splitting bands")
    low, high = _split_bands(audio, bass, fs)

    say(0.18, "Loading HRTFs")
    irs, ir_fs, pos = _load_sofa()
    el, az = pos[:, 1], np.mod(pos[:, 0], 360.0)
    keep = np.abs(el) <= 2.0
    irs, az = irs[keep], az[keep]
    az, idx = np.unique(np.round(az, 3), return_index=True)
    irs = irs[idx]
    irs = _resample_last(irs, ir_fs, fs)

    block = 4096
    hop = block // 2
    n_ir = irs.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(block + n_ir - 1)))
    spec = np.fft.rfft(irs, n=nfft, axis=-1)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(block) / block)

    n = high.shape[0]
    nb = int(np.ceil(n / hop)) + 1
    padded = np.pad(high, ((0, nb * hop + block), (0, 0)))
    out = np.zeros((padded.shape[0] + nfft, 2))
    az_ext = np.concatenate([az, [az[0] + 360.0]])

    say(0.22, "Orbiting")
    for b in range(nb):
        s0 = b * hop
        chunk = padded[s0:s0 + block]
        if not chunk.any():
            continue
        t = (s0 + block / 2) / fs
        a = (360.0 * t / period) % 360.0
        j = min(max(np.searchsorted(az_ext, a, side="right") - 1, 0), len(az) - 1)
        lo, hi = az_ext[j], az_ext[j + 1]
        fr = 0.0 if hi == lo else (a - lo) / (hi - lo)
        sp = (1 - fr) * spec[j] + fr * spec[(j + 1) % len(az)]
        m = np.fft.rfft(chunk.mean(axis=1) * win, n=nfft)
        out[s0:s0 + nfft] += np.fft.irfft(m[None, :] * sp, n=nfft, axis=-1).T
        if b % 150 == 0:
            say(0.22 + 0.68 * b / nb, "Orbiting")

    out = out[:n]
    say(0.92, "Reverb and bass")
    out = _reverb(out, fs, reverb)
    out[:, 0] += low
    out[:, 1] += low
    peak = np.abs(out).max()
    if peak > 0.99:
        out *= 0.99 / peak
    return out


# ------------------------------------------------------ ambisonic pipeline

def _virtual_speakers():
    dirs = [(a, 0.0) for a in range(0, 360, 45)]
    for a in (45, 135, 225, 315):
        dirs += [(a, 35.0), (a, -35.0)]
    v = []
    for a, e in dirs:
        ar, er = np.radians(a), np.radians(e)
        v.append((np.cos(ar) * np.cos(er), np.sin(ar) * np.cos(er), np.sin(er)))
    return np.array(v)


def _ambi_filters(fs, width=1.0):
    """Fold a 16-point virtual array into 4 fixed filters per ear."""
    irs, ir_fs, pos = _load_sofa()
    irs = _resample_last(irs, ir_fs, fs)
    az, el = np.radians(pos[:, 0]), np.radians(pos[:, 1])
    vecs = np.stack([np.cos(az) * np.cos(el),
                     np.sin(az) * np.cos(el),
                     np.sin(el)], axis=1)
    spk = _virtual_speakers()
    picked = irs[[int(np.argmax(vecs @ s)) for s in spk]]
    g = 2.0 / len(spk)
    filt = np.zeros((4, 2, picked.shape[-1]))
    for i in range(len(spk)):
        filt[0] += g * picked[i]
        for k in range(3):
            filt[k + 1] += g * width * spk[i, k] * picked[i]
    return filt


def process_ambisonic(b, fs, period=14.0, width=1.0, progress=None):
    def say(p, m):
        if progress:
            progress(p, m)

    say(0.12, "Building binaural filters")
    filt = _ambi_filters(fs, width)

    # ACN order is W, Y, Z, X.
    W, Y, Z, X = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    say(0.20, "Rotating sound field")
    # Per-sample yaw. Nothing is blocked, so there is nothing to click.
    th = 2 * np.pi * np.arange(len(W)) / (period * fs)
    c, s = np.cos(th), np.sin(th)
    chans = (W, X * c - Y * s, X * s + Y * c, Z)

    out = np.zeros((len(W), 2))
    for ci, sig in enumerate(chans):
        for ear in (0, 1):
            out[:, ear] += _fft_conv(sig, filt[ci, ear])
        say(0.25 + 0.65 * (ci + 1) / 4, f"Rendering {'WXYZ'[ci]}")

    peak = np.abs(out).max()
    if peak > 0:
        out *= 0.97 / peak
    return out


# ------------------------------------------------------------------ output

def write_outputs(samples, fs, outdir: Path, stem: str,
                  source_video: str | None = None, progress=None):
    outdir.mkdir(parents=True, exist_ok=True)
    wav = outdir / f"{stem}_8d.wav"
    sf.write(wav, samples, int(fs), subtype="PCM_24")

    mp3 = outdir / f"{stem}_8d.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav),
                    "-b:a", "320k", str(mp3)], check=True, capture_output=True)

    made = [wav, mp3]
    if source_video:
        if progress:
            progress(0.96, "Muxing video")
        mp4 = outdir / f"{stem}_8d.mp4"
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source_video), "-i", str(wav),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "320k", "-shortest", str(mp4)],
            capture_output=True, text=True)
        if r.returncode == 0 and mp4.exists():
            made.append(mp4)
    return made
