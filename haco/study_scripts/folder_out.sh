#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%m%d%Y%H%M)"   # mmddyyyyhhmm

for d in audio_recordings trajectory_data videos; do
  dest="${d}_${ts}"
  mkdir -p "$dest"
  for f in "$d"/*; do
    [ -f "$f" ] || continue
    mv "$f" "$dest"/
  done
done
