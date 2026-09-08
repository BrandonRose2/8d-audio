"""Synthesize a short royalty-free demo track to test 8D processing."""
import numpy as np, soundfile as sf

FS = 44100
BPM = 92
beat = 60.0 / BPM
bar = 4 * beat

def note(f, dur, kind="pluck", amp=0.3):
    n = int(dur * FS); t = np.arange(n) / FS
    if kind == "pluck":
        env = np.exp(-t * 4.5)
        w = (np.sin(2*np.pi*f*t) + 0.4*np.sin(4*np.pi*f*t)
             + 0.2*np.sin(6*np.pi*f*t) + 0.1*np.sin(8*np.pi*f*t))
    elif kind == "pad":
        att = np.clip(t / 0.35, 0, 1); rel = np.clip((dur - t) / 0.5, 0, 1)
        env = att * rel
        det = 1.003
        w = (np.sin(2*np.pi*f*t) + np.sin(2*np.pi*f*det*t)
             + np.sin(2*np.pi*f/det*t) + 0.5*np.sin(4*np.pi*f*t))
    else:  # bass
        env = np.exp(-t * 1.6)
        w = np.sin(2*np.pi*f*t) + 0.25*np.sin(4*np.pi*f*t)
    return amp * env * w

def hz(semi):  # semitones from A2 = 110 Hz
    return 110.0 * 2 ** (semi / 12.0)

# Am - F - C - G, two times through
prog = [
    (0,  [0, 3, 7, 12]),      # Am
    (-4, [0, 4, 7, 12]),      # F
    (3,  [0, 4, 7, 12]),      # C
    (10, [0, 4, 7, 12]),      # G
] * 2

total = int(len(prog) * bar * FS) + FS
mix = np.zeros(total)

for i, (root, shape) in enumerate(prog):
    t0 = i * bar

    # pad, one octave up
    for s in shape:
        s0 = int(t0 * FS)
        v = note(hz(root + s + 12), bar * 0.98, "pad", 0.055)
        mix[s0:s0+len(v)] += v

    # bass on beats 1 and 3
    for b in (0, 2):
        s0 = int((t0 + b * beat) * FS)
        v = note(hz(root - 12), beat * 1.9, "bass", 0.42)
        mix[s0:s0+len(v)] += v

    # arpeggio, eighth notes, two octaves up
    for e in range(8):
        s0 = int((t0 + e * beat / 2) * FS)
        deg = shape[e % len(shape)] + (12 if e >= 4 else 24)
        v = note(hz(root + deg), beat * 0.9, "pluck", 0.10)
        mix[s0:s0+len(v)] += v

    # hats on offbeat eighths
    rng = np.random.default_rng(1000 + i)
    for e in range(8):
        if e % 2 == 0:
            continue
        s0 = int((t0 + e * beat / 2) * FS)
        n = int(0.05 * FS); tt = np.arange(n) / FS
        h = rng.normal(0, 1, n) * np.exp(-tt * 90) * 0.05
        h = h - np.convolve(h, np.ones(9)/9, mode="same")   # crude highpass
        mix[s0:s0+n] += h

    # kick on 1 and 3
    for b in (0, 2):
        s0 = int((t0 + b * beat) * FS)
        n = int(0.28 * FS); tt = np.arange(n) / FS
        f = 115 * np.exp(-tt * 28) + 46
        k = np.sin(2*np.pi*np.cumsum(f)/FS) * np.exp(-tt * 8) * 0.55
        mix[s0:s0+n] += k

mix /= np.abs(mix).max() / 0.85
stereo = np.stack([mix, mix], axis=1)
sf.write("demo.wav", stereo, FS, subtype="PCM_24")
print(f"wrote demo.wav  {len(mix)/FS:.1f}s")
