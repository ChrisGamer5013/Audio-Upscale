"""Self-update niceties: latest-release check and starter model fetch."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from . import __version__

REPO = "ChrisGamer5013/Audio-Upscale"
LATEST_API = f"https://api.github.com/repos/{REPO}/releases/latest"
STARTER_MODEL_NAME = "starter_bwe_48k.onnx"
STARTER_MODEL_URL = (
    f"https://github.com/{REPO}/releases/latest/download/{STARTER_MODEL_NAME}"
)


def check_for_update(timeout: float = 4.0) -> tuple[str, str | None]:
    """Returns (current, latest_or_None, url). Never raises."""
    try:
        req = urllib.request.Request(
            LATEST_API, headers={"User-Agent": f"audio-upscaler/{__version__}"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            tag = json.loads(r.read().decode()).get("tag_name", "")
        latest = tag.lstrip("vV")
        return __version__, (latest if latest != __version__ else None), tag
    except Exception:
        return __version__, None, ""


def models_dir() -> Path:
    from .bootstrap import app_dir

    d = app_dir() / "models"
    d.mkdir(exist_ok=True)
    return d


def starter_model_path() -> Path:
    return models_dir() / STARTER_MODEL_NAME


def ensure_starter_model(log=print) -> Path | None:
    """Download the bundled-size starter BWE model from the latest release."""
    p = starter_model_path()
    if p.is_file() and p.stat().st_size > 10_000:
        return p
    try:
        log(f"Downloading starter model from {STARTER_MODEL_URL} ...")
        urllib.request.urlretrieve(STARTER_MODEL_URL, p)
        if p.is_file() and p.stat().st_size > 10_000:
            log(f"  -> {p}")
            return p
    except Exception as exc:
        log(f"  starter model download failed: {exc}")
    return None
