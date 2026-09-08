# 8D Audio

Turn a track or a recording into "8D audio" — a binaural render that orbits the
listener's head — using measured head-related transfer functions rather than
simple volume panning.

Drop a file into the web UI, or use the CLI. Everything runs locally.

![local web app](https://img.shields.io/badge/runs-locally-brightgreen) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

## What "8D audio" actually is

Not a format, and there is no eighth dimension. It is a stereo file whose
content has been spatialized with a binaural panner that slowly circles the
listener, usually with reverb added for a sense of room. **It only works on
headphones** — on speakers it collapses back to ordinary stereo.

Most 8D uploads are a plain LFO auto-pan, which is why they sound like someone
turning a volume knob. This project does two better things.

## Two processing paths

### Rotate the sound field (best, macOS only)

If the file carries a 4-channel Apple APAC spatial-audio track — which every
recent iPhone records by default — the app rotates the **actual recorded sound
field** rather than simulating movement. The room, its reverb and the height
information all turn together, the way they would if you spun your chair.

The track is first-order ambisonic in ACN/SN3D. Because ffmpeg has no APAC
decoder (its `apac` is Marian's unrelated codec), decoding goes through
CoreAudio via a small Swift/AVFoundation helper, which is why this path needs
macOS.

Implementation notes:

- A 16-point virtual loudspeaker array (8 horizontal, 4 up, 4 down) is folded
  into **four fixed filters per ear**. Decoding is linear, so 16 speaker
  convolutions collapse into 4 that never change.
- Because those filters are constant, rotation reduces to a 2×2 matrix on the
  X and Y channels applied **per sample**. There is no block processing in the
  rotation at all, so there is nothing to click.

### Orbit a mono sum (works on anything)

For ordinary files, a mono sum is panned around the head with KEMAR HRTFs,
interpolating continuously between measured angles.

- **Bass stays put.** Below the crossover (default 120 Hz) audio is summed to
  mono and left centered. Orbiting sub-bass does not read as spatial, it reads
  as broken, and it wrecks the low end.
- The crossover is `high = audio - lowpass(audio)` with a linear-phase FIR, so
  the bands sum back to the original exactly — none of the phase error of
  cascaded IIR filters.
- The orbit uses weighted overlap-add with a periodic Hann window at 50%
  overlap, which sums to exactly 1.0, so the filter can change every block
  without clicks or level ripple.

## Install

Requires Python 3.11+, ffmpeg, and (for the spatial path) macOS with Xcode
command line tools.

```bash
git clone https://github.com/BrandonRose2/8d-audio.git
cd 8d-audio
./setup.sh
```

`setup.sh` creates the virtualenv, installs dependencies, downloads the MIT
KEMAR HRTF set, and builds the Swift decoder when possible.

## Use

```bash
./start
```

Opens <http://127.0.0.1:8765>. Drop a file, pick a rotation speed, download WAV
/ MP3, or an MP4 with the video stream copied through and the new audio
attached.

### CLI

```bash
# any file
./.venv/bin/python cli/make8d.py song.wav -s app/kemar.sofa --period 12

# a 4-channel ambisonic wav
./.venv/bin/python cli/spin_ambi.py bformat.wav -s app/kemar.sofa -o out.wav
```

## Settings that matter

| Setting | Default | Notes |
|---|---|---|
| Rotation speed | 12 s | Seconds per circle. Under 6 s tends to feel queasy. |
| Bass crossover | 120 Hz | Raise it for bass-heavy material. A dance track with 65% of its energy under 150 Hz wants 150–160. |
| Reverb | 25% | Drop to 10–15% on a finished master or a live-sounding room; it already has its own space. |
| Directional focus | 1.0 | Ambisonic path only. Higher is sharper, lower is more diffuse. |

**Source quality dominates the result.** HRTF front/back cues live in the
4–16 kHz band. A dull or heavily compressed recording will swing left to right
but never really sound *behind* you, because there is nothing up there to
filter. A full-bandwidth master is a completely different experience.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8765` | Listen port. |
| `HOST` | `127.0.0.1` | Bind address. Set a Tailscale IP for remote access. |
| `EIGHTD_TOKEN` | unset | Shared secret. When set, every request needs `X-Access-Token` or `?t=`. |
| `EIGHTD_MAX_UPLOAD_MB` | unset | Reject uploads above this size. |

## Deploying

This is built to run locally, and that is where it works best. Before putting it
on a network, know:

1. **The spatial-audio path is macOS-only.** The decoder is a Mach-O binary
   built against AVFoundation. On Linux the app detects its absence and offers
   only the stereo path.
2. **There is no user model.** `EIGHTD_TOKEN` is a single shared secret, not
   accounts. Without it, an exposed instance is an open compute endpoint.
3. **It is CPU-bound and stateful.** Jobs run in-process against local disk, one
   at a time. Serverless platforms are a poor fit — request body caps and
   execution limits rule out large files.
4. **Uploads are other people's recordings.** Anything you host, you are
   hosting on their behalf.

A container is provided for a small always-on box (Fly.io, Railway, Render, a
NAS). It is stereo-path only:

```bash
docker build -t 8d-audio .
docker run -p 8765:8765 -e EIGHTD_TOKEN=pick-a-long-secret \
  -e EIGHTD_MAX_UPLOAD_MB=200 -v "$PWD/data:/app/app/jobs" 8d-audio
```

To keep full functionality *and* reach it remotely, expose the Mac instance over
a private network instead of deploying a copy. With Tailscale, either

```bash
tailscale serve --bg 8765          # HTTPS on your tailnet, app stays on localhost
```

or bind the tailnet address directly:

```bash
EIGHTD_TOKEN=$(openssl rand -hex 24) HOST=$(tailscale ip -4) ./start
```

`start` refuses a non-localhost bind without a token unless you confirm.

## Credits

HRTF measurements are the MIT KEMAR set by Bill Gardner and Keith Martin at the
MIT Media Lab, via the [SOFA conventions
database](https://www.sofaconventions.org/). The data is downloaded at setup
time and is not redistributed in this repository.

## License

MIT, see [LICENSE](LICENSE). The KEMAR data is covered by its own terms.
