#!/usr/bin/env bash
# One-time setup: virtualenv, Python deps, HRTF data, and (on macOS) the
# Apple Spatial Audio decoder.
set -euo pipefail
cd "$(dirname "$0")"

command -v ffmpeg >/dev/null || {
  echo "ffmpeg is required."
  echo "  macOS:  brew install ffmpeg"
  echo "  Debian: sudo apt install ffmpeg"
  exit 1
}

echo "==> Python environment"
python3 -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt

echo "==> HRTF data (MIT KEMAR)"
if [ ! -f app/kemar.sofa ]; then
  curl -fL --retry 3 -o app/kemar.sofa \
    https://sofacoustics.org/data/database/mit/mit_kemar_normal_pinna.sofa
  echo "    downloaded $(du -h app/kemar.sofa | cut -f1)"
else
  echo "    already present"
fi

echo "==> Apple Spatial Audio decoder"
if [ "$(uname -s)" = "Darwin" ] && command -v swiftc >/dev/null; then
  mkdir -p app/bin
  swiftc -O app/src/decode_track.swift -o app/bin/decode_track 2>/dev/null
  echo "    built app/bin/decode_track"
else
  echo "    skipped: needs macOS + swiftc."
  echo "    The app still runs; the spatial-audio path is simply unavailable."
fi

echo
echo "Done. Start it with:  ./start"
