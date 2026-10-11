"""Audio Upscaler HD - entry point.

Usage:
    python main.py            -> launch the GUI
    python main.py gui        -> launch the GUI
    python main.py cli --help -> command line interface
"""
from __future__ import annotations

import sys


def main() -> int:
    args = sys.argv[1:]
    if args and args[0].lower() == "cli":
        from audio_upscaler.cli import run_cli

        return run_cli(args[1:])

    # default: GUI
    try:
        from audio_upscaler.gui.app import launch
    except ImportError as exc:  # pragma: no cover - depends on env
        print(f"GUI dependencies missing ({exc}).")
        print("Install them with:  pip install customtkinter Pillow tkinterdnd2")
        print("Or use the CLI:     python main.py cli --help")
        return 1

    launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
