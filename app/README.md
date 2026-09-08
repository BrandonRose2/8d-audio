# 8D Audio (local web app)

    ~/8d/start

Opens http://127.0.0.1:8765. Bound to localhost only; files never leave the Mac.

## What it does

Drop an audio or video file. Two processing paths:

- **Rotate sound field** — used automatically when the file carries a 4-channel
  Apple APAC spatial-audio track (iPhone recordings). Rotates the actual
  recorded field, so the room, its reverb, and height all turn together.
  Decoded through CoreAudio via `bin/decode_track`, because ffmpeg has no APAC
  decoder.
- **Orbit a mono sum** — the fallback for ordinary files. Pans a mono sum around
  the head with KEMAR HRTFs, keeping bass mono and centered.

Outputs land in `jobs/<id>/`: 24-bit WAV, 320k MP3, and optionally an MP4 with
the video stream copied and the new audio attached.

## Layout

    start              launcher
    app/server.py      FastAPI routes
    app/engine.py      all DSP
    app/static/        single-file UI
    app/kemar.sofa     MIT KEMAR HRTF set
    app/bin/decode_track   Swift/AVFoundation APAC decoder
    app/jobs/          uploads and renders, safe to delete

## Notes

- Files remuxed or re-encoded after recording usually lose the metadata
  CoreAudio needs for APAC. The app detects this and falls back to the stereo
  path with a note rather than failing.
- Job state is in memory, so progress is lost on restart, but rendered files
  stay downloadable because `/file` reads from disk.
- Delete old work with `rm -rf ~/8d/app/jobs/*`.
