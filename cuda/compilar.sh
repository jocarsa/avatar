#!/usr/bin/env bash
set -euo pipefail

SRC="cuda_fps_mlt_single.cu"
OUT="cuda_fps_mlt"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found." >&2
  exit 1
fi

if ! command -v pkg-config >/dev/null 2>&1; then
  echo "ERROR: pkg-config not found." >&2
  exit 1
fi

if ! pkg-config --exists opencv4; then
  echo "ERROR: opencv4 not found via pkg-config." >&2
  echo "Ubuntu/Debian: sudo apt-get install -y libopencv-dev pkg-config" >&2
  exit 1
fi

# Ensure X11 dev is present (headers + link)
if ! ldconfig -p 2>/dev/null | grep -q "libX11.so"; then
  echo "ERROR: libX11 not found on this system (runtime). Install it." >&2
  echo "Ubuntu/Debian: sudo apt-get install -y libx11-6 libx11-dev" >&2
  exit 1
fi

CFLAGS="$(pkg-config --cflags opencv4)"
LIBS="$(pkg-config --libs opencv4)"

echo "Compiling $SRC -> $OUT"

# IMPORTANT: -lX11 must be on the final nvcc link line.
nvcc -O3 -std=c++17 --use_fast_math \
  $CFLAGS \
  "$SRC" \
  -o "$OUT" \
  $LIBS -lX11

echo "Done."
echo "Run: ./$OUT your_model.obj"

