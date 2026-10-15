"""Persist GUI settings + queue to %APPDATA%/AudioUpscalerHD/settings.json."""
from __future__ import annotations

import json
import os
from pathlib import Path


def _store_path() -> Path:
    base = Path(os.environ.get("APPDATA", ".")) / "AudioUpscalerHD"
    base.mkdir(parents=True, exist_ok=True)
    return base / "settings.json"


def load_settings() -> dict:
    p = _store_path()
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_settings(data: dict) -> None:
    try:
        _store_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass  # persistence must never break the app
