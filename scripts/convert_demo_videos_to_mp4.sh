#!/usr/bin/env bash
set -euo pipefail

# Convert one MPG file or a whole folder of videos into MP4 for browser preview.
#
# Usage:
#   scripts/convert_demo_videos_to_mp4.sh --input data/s1_processed --output data/demo_mp4/s1_processed
#   scripts/convert_demo_videos_to_mp4.sh --input data/s1_processed/bbal9a.mpg --output data/demo_mp4/s1_processed
#
# Notes:
# - Keeps the same base filename, only extension becomes .mp4.
# - Uses H.264 + AAC for wide browser support.

INPUT=""
OUTPUT_DIR="data/demo_mp4"
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "Missing required --input"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is not installed or not in PATH."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

convert_file() {
  local src="$1"
  local base out
  base="$(basename "$src")"
  base="${base%.*}"
  out="${OUTPUT_DIR}/${base}.mp4"

  if [[ -f "$out" && "$OVERWRITE" -ne 1 ]]; then
    echo "skip  $src (exists: $out)"
    return 0
  fi

  echo "conv  $src -> $out"
  ffmpeg -hide_banner -loglevel error -y \
    -i "$src" \
    -c:v libx264 -preset veryfast -crf 22 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    -c:a aac -b:a 128k \
    "$out"
}

if [[ -f "$INPUT" ]]; then
  convert_file "$INPUT"
elif [[ -d "$INPUT" ]]; then
  shopt -s nullglob
  files=("$INPUT"/*.mpg "$INPUT"/*.mpeg "$INPUT"/*.avi "$INPUT"/*.mov "$INPUT"/*.webm "$INPUT"/*.mp4)
  shopt -u nullglob

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No video files found in: $INPUT"
    exit 1
  fi

  for f in "${files[@]}"; do
    convert_file "$f"
  done
else
  echo "Input does not exist: $INPUT"
  exit 1
fi

echo "Done. MP4 files are in: $OUTPUT_DIR"
