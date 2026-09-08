#!/usr/bin/env python3
"""
spin_ambi.py - rotate a first-order ambisonic recording and render it binaural.

Unlike make8d.py, which fakes an orbit by panning a mono sum, this rotates the
actual recorded sound field. The room, the reverb and the height information all
turn together, the way they would if you spun your chair.

Input is 4-channel ACN/SN3D B-format (what iPhone Spatial Audio captures).
"""

import argparse
import sys

import h5py
import numpy as np
import soundfile as sf


def load_sofa_all(path):
    """All measurements: irs (M,2,N), fs, unit vectors (M,3) in x,y,z."""
    with h5py.File(path, "r") as f:
        ir = np.asarray(f["Data.IR"], dtype=np.float64)[:, :2, :]
        fs = float(np.asarray(f["Data.SamplingRate"]).flatten()[0])
        pos = np.asarray(f["SourcePosition"], dtype=np.float64)
    az, el = np.radians(pos[:, 0]), np.radians(pos[:, 1])
    vecs = np.stack([np.cos(az) * np.cos(el),      # x, front
                     np.sin(az) * np.cos(el),      # y, left
                     np.sin(el)], axis=1)          # z, up
    return ir, fs, vecs


def resample_last(x, fs_in, fs_out):
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


def virtual_speakers():
    """A 16-point pseudo-uniform layout: 8 horizontal, 4 up, 4 down."""
    dirs = []
    for a in range(0, 360, 45):
        dirs.append((a, 0.0))
    for a in (45, 135, 225, 315):
        dirs.append((a, 35.0))
        dirs.append((a, -35.0))
    out = []
    for a, e in dirs:
        ar, er = np.radians(a), np.radians(e)
        out.append((np.cos(ar) * np.cos(er), np.sin(ar) * np.cos(er), np.sin(er)))
    return np.array(out)


def ambi_to_binaural_filters(sofa_path, fs, weight=1.0):
    """Collapse a virtual-speaker array into four fixed filters per ear.

    Decoding is linear, so instead of convolving 16 speaker feeds we fold the
    array into one filter per B-format channel. Only 8 convolutions total, and
    they never change - the rotation happens on X/Y before this stage.
    """
    irs, ir_fs, vecs = load_sofa_all(sofa_path)
    irs = resample_last(irs, ir_fs, fs)
    spk = virtual_speakers()
    n_spk = len(spk)

    # Nearest measured direction for each virtual speaker.
    idx = [int(np.argmax(vecs @ s)) for s in spk]
    picked = irs[idx]                                   # (n_spk, 2, N)
    err = [np.degrees(np.arccos(np.clip(vecs[i] @ s, -1, 1)))
           for i, s in zip(idx, spk)]
    print(f"  virtual array: {n_spk} points, worst HRTF match {max(err):.1f} deg",
          file=sys.stderr)

    n = picked.shape[-1]
    # In-phase (cardioid) order-1 decode: gain = 1 + weight * cos(angle).
    # Non-negative, so no polarity artifacts.
    g = 2.0 / n_spk
    filt = np.zeros((4, 2, n))                          # [W,X,Y,Z] x [L,R] x taps
    for i in range(n_spk):
        filt[0] += g * picked[i]
        filt[1] += g * weight * spk[i, 0] * picked[i]
        filt[2] += g * weight * spk[i, 1] * picked[i]
        filt[3] += g * weight * spk[i, 2] * picked[i]
    return filt


def fft_convolve_add(x, h, block=1 << 16):
    """Overlap-add FFT convolution, keeps memory flat on long files."""
    n_h = len(h)
    nfft = 1 << int(np.ceil(np.log2(block + n_h - 1)))
    H = np.fft.rfft(h, n=nfft)
    out = np.zeros(len(x) + n_h)
    for s in range(0, len(x), block):
        seg = x[s:s + block]
        y = np.fft.irfft(np.fft.rfft(seg, n=nfft) * H, n=nfft)
        out[s:s + len(y)] += y[:len(out) - s]
    return out[:len(x)]


def main():
    p = argparse.ArgumentParser(description="Rotate an ambisonic recording, render binaural.")
    p.add_argument("input", help="4-channel ACN/SN3D wav, or a .raw float32 with --raw")
    p.add_argument("-s", "--sofa", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--period", type=float, default=14.0, help="seconds per rotation")
    p.add_argument("--raw-fs", type=float, default=0.0,
                   help="treat input as headerless float32 4ch at this rate")
    p.add_argument("--width", type=float, default=1.0,
                   help="directional weight 0..1.5 (1.0 = cardioid)")
    p.add_argument("--preview", type=float, default=0.0)
    args = p.parse_args()

    if args.raw_fs > 0:
        fs = args.raw_fs
        b = np.fromfile(args.input, dtype=np.float32).reshape(-1, 4).astype(np.float64)
    else:
        b, fs_i = sf.read(args.input, always_2d=True, dtype="float64")
        fs = float(fs_i)
    if b.shape[1] != 4:
        raise SystemExit(f"need 4 ambisonic channels, got {b.shape[1]}")
    if args.preview > 0:
        b = b[:int(args.preview * fs)]
    print(f"  {b.shape[0]/fs:.1f}s of B-format @ {fs:.0f} Hz", file=sys.stderr)

    # ACN order is W, Y, Z, X. Reorder to W, X, Y, Z.
    W, Y, Z, X = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    print("Building ambisonic-to-binaural filters", file=sys.stderr)
    filt = ambi_to_binaural_filters(args.sofa, fs, weight=args.width)

    # Continuous per-sample yaw. No blocking, so no crossfade artifacts at all.
    th = 2 * np.pi * np.arange(len(W)) / (args.period * fs)
    c, s = np.cos(th), np.sin(th)
    Xr = X * c - Y * s
    Yr = X * s + Y * c

    print(f"Rendering, one rotation every {args.period:.1f}s", file=sys.stderr)
    chans = (W, Xr, Yr, Z)
    out = np.zeros((len(W), 2))
    for ci, sig in enumerate(chans):
        for ear in (0, 1):
            out[:, ear] += fft_convolve_add(sig, filt[ci, ear])
        print(f"  channel {'WXYZ'[ci]} done", file=sys.stderr)

    peak = np.abs(out).max()
    if peak > 0:
        out *= 0.97 / peak
    sf.write(args.out, out, int(fs), subtype="PCM_24")
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
