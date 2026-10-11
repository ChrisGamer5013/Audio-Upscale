"""Ensure ffmpeg is available (thin wrapper around audio_upscaler.bootstrap).

Run directly:  python scripts/install_ffmpeg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_upscaler.bootstrap import ensure_ffmpeg

if __name__ == "__main__":
    sys.exit(0 if ensure_ffmpeg() else 1)
