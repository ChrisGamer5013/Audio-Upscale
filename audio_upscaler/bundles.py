"""Multi-model bundle manifests (model.json / manifest.json / config.json).

Many ONNX releases ship several files that only work together (encoder,
decoder, vocoder, denoiser, pitch extractor, ...).  A small JSON manifest
tells us which files to run, in which order, and with which options:

    {
      "name": "voice-v1",
      "sample_rate": 48000,
      "stages": [
        {"file": "encoder.onnx",  "role": "encoder"},
        {"file": "decoder.onnx",  "kind": "mel2stft"},
        {"file": "vocoder.onnx",  "role": "vocoder"}
      ]
    }

Recognised manifest keys (first non-empty wins):
    stages / pipeline / chain / models / components / files

Each entry may be a plain path string, or an object using one of
``file`` / ``path`` / ``name`` / ``model`` / ``onnx`` / ``filename`` and
optionally ``role``, ``kind`` and ``sidecar`` (a params object).

Folders are accepted too: a directory containing one of the manifest names
above is loaded, and a directory with exactly one ``*.onnx`` is treated as a
single-model bundle (sherpa-style model dirs).  This also understands
nested layouts such as ``{"model": {"files": [...]}}`` used by model
packages like SpeechT5 (encoder + decoder + vocoder + tokenizer + speaker).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_NAMES = ("model.json", "manifest.json", "bundle.json",
                  "config.json", "model-manifest.json")
_STAGE_KEYS = ("stages", "pipeline", "chain", "models", "components", "files")
_ENTRY_PATH_KEYS = ("file", "path", "name", "model", "onnx", "filename")
_OPTION_KEYS = ("kind", "sample_rate", "sr", "n_fft", "hop", "win", "n_mels",
                "fmin", "fmax", "log", "top_db", "gl_iters", "chunk_frames",
                "overlap_frames", "power")
_MODEL_SUFFIXES = (".onnx",)


@dataclass
class BundleStage:
    path: Path
    role: str = ""
    kind: str = ""
    sidecar: dict = field(default_factory=dict)


@dataclass
class Bundle:
    name: str
    stages: list[BundleStage]
    options: dict = field(default_factory=dict)
    manifest: Path | None = None


@dataclass
class ModelSpec:
    """Expanded model selection: ordered paths + bundle-level overrides."""

    paths: list[Path]
    options: dict = field(default_factory=dict)
    stage_meta: list[dict] = field(default_factory=list)
    bundle_name: str = ""
    manifest: Path | None = None


def split_spec(raw) -> list[str]:
    """Split 'a.onnx+b.onnx, c.onnx;d.onnx' into stripped parts."""
    return [p.strip() for p in re.split(r"[+;,]", str(raw)) if p and p.strip()]


def _looks_like_manifest(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".json"


def _entry_path(entry, base: Path) -> Path | None:
    if isinstance(entry, str):
        raw = entry.strip()
    elif isinstance(entry, dict):
        raw = ""
        for key in _ENTRY_PATH_KEYS:
            val = entry.get(key)
            if isinstance(val, str) and val.strip():
                raw = val.strip()
                break
        if not raw:
            return None
    else:
        return None
    if Path(raw).suffix.lower() in (".txt", ".json", ".model", ".npy", ".bin",
                                    ".vocab", ".yaml", ".yml"):
        return None  # tokenizers, vocabularies, speaker embeddings, configs
    p = Path(raw)
    if not p.is_absolute():
        p = base / p
    return p


def _as_entries(val) -> list | None:
    """Normalize a stage container: list of entries or {name: path|object}."""
    if isinstance(val, list):
        return val or None
    if not isinstance(val, dict) or not val:
        return None
    if any(isinstance(val.get(k), str) for k in _ENTRY_PATH_KEYS):
        return [val]  # a single stage object, not a mapping
    entries: list = []
    for name, item in val.items():
        if isinstance(item, str):
            entries.append({"file": item, "role": str(name)})
        elif isinstance(item, dict) and any(
                isinstance(item.get(k), str) for k in _ENTRY_PATH_KEYS):
            entry = dict(item)
            entry.setdefault("role", str(name))
            entries.append(entry)
        else:
            return None  # nested config object, not a stage mapping
    return entries or None


def _find_entries(data: dict) -> list | None:
    for key in _STAGE_KEYS:
        entries = _as_entries(data.get(key))
        if entries:
            return entries
        val = data.get(key)
        if isinstance(val, dict):
            for key2 in _STAGE_KEYS:
                entries = _as_entries(val.get(key2))
                if entries:
                    return entries
    # generic one-level nesting: {"model": {"files": [...]}} (model packages)
    for val in data.values():
        if isinstance(val, dict):
            for key2 in _STAGE_KEYS:
                entries = _as_entries(val.get(key2))
                if entries:
                    return entries
    return None


def load_bundle(path: str | Path) -> Bundle | None:
    """Load a manifest file or folder bundle.

    Returns ``None`` when *path* is a plain model file (not a bundle).
    Raises ``RuntimeError`` for unreadable/incomplete manifests.
    """
    p = Path(path)
    manifest: Path | None = None
    if p.is_dir():
        for name in MANIFEST_NAMES:
            cand = p / name
            if cand.is_file():
                manifest = cand
                break
        if manifest is None:
            models = sorted(q for q in p.glob("*.onnx") if q.is_file())
            if len(models) == 1:
                return Bundle(name=p.name,
                              stages=[BundleStage(models[0])])
            return None
    elif _looks_like_manifest(p):
        manifest = p
    else:
        return None

    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Unreadable bundle manifest '{manifest.name}': {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"'{manifest.name}': expected a JSON object.")

    entries = _find_entries(data)
    if not entries:
        raise RuntimeError(
            f"'{manifest.name}': no model list found (expected one of: "
            f"{', '.join(_STAGE_KEYS)}).")

    base = manifest.parent
    stages: list[BundleStage] = []
    missing: list[str] = []
    unsupported: list[str] = []
    for entry in entries:
        fp = _entry_path(entry, base)
        if fp is None:
            continue
        suffix = fp.suffix.lower()
        if suffix in (".keras", ".h5"):
            unsupported.append(fp.name)
            continue
        if suffix != ".onnx":
            continue  # tokenizer/vocab/config, not a runnable stage
        if not fp.exists():
            missing.append(fp.name)
            continue
        meta = entry if isinstance(entry, dict) else {}
        stages.append(BundleStage(
            path=fp,
            role=str(meta.get("role") or meta.get("type") or ""),
            kind=str(meta.get("kind") or ""),
            sidecar=dict(meta.get("sidecar") or {}),
        ))
    if missing:
        raise RuntimeError(
            f"'{manifest.name}': missing model file(s): {', '.join(missing)}")
    if unsupported:
        raise RuntimeError(
            f"'{manifest.name}': .keras/.h5 stages are not supported in "
            f"bundles ({', '.join(unsupported)}); convert them with "
            "scripts\\keras_to_onnx.py first.")
    if not stages:
        raise RuntimeError(
            f"'{manifest.name}': no usable model entries "
            f"({', '.join(_MODEL_SUFFIXES)}).")

    options: dict = {}
    for key in _OPTION_KEYS:
        val = data.get(key)
        if not isinstance(val, (dict, list)) and val is not None:
            options[key] = val
    for key in ("sidecar", "params", "config"):
        if isinstance(data.get(key), dict):
            options.update(data[key])
    return Bundle(name=str(data.get("name") or manifest.stem),
                  stages=stages, options=options, manifest=manifest)


def expand_model_spec(raw) -> ModelSpec:
    """Expand a model spec (chain text, manifest, folder) into ordered paths."""
    parts = split_spec(raw)
    spec = ModelSpec(paths=[], options={}, stage_meta=[])
    for part in parts:
        p = Path(part)
        bundle: Bundle | None = None
        if p.is_dir() or _looks_like_manifest(p):
            bundle = load_bundle(p)
            if bundle is None and p.is_dir():
                models = sorted(q for q in p.glob("*.onnx") if q.is_file())
                raise RuntimeError(
                    f"Folder '{p.name}' contains {len(models)} .onnx file(s); "
                    "add a model.json describing the order (see "
                    "models/README.md) or chain the files with '+'.")
        if bundle is not None:
            if not spec.bundle_name:
                spec.bundle_name = bundle.name
                spec.manifest = bundle.manifest
            spec.options.update(bundle.options)
            for stage in bundle.stages:
                spec.paths.append(stage.path)
                meta: dict = {}
                if stage.role:
                    meta["role"] = stage.role
                if stage.kind:
                    meta["kind"] = stage.kind
                if stage.sidecar:
                    meta["sidecar"] = stage.sidecar
                spec.stage_meta.append(meta)
        else:
            spec.paths.append(p)
            spec.stage_meta.append({})
    return spec


def manifest_template() -> str:
    """Starter JSON shown by the GUI 'Bundle' helper."""
    return json.dumps({
        "name": "my-bundle",
        "sample_rate": 48000,
        "stages": [
            {"file": "stage1.onnx", "role": "encoder"},
            {"file": "stage2.onnx", "role": "decoder"},
            {"file": "stage3.onnx", "role": "vocoder"},
        ],
    }, indent=2)


__all__ = ["Bundle", "BundleStage", "ModelSpec", "MANIFEST_NAMES",
           "expand_model_spec", "load_bundle", "manifest_template", "split_spec"]
