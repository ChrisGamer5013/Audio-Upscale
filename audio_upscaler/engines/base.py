"""Engine interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..presets import Preset

ProgressCb = Callable[[float, str], None]


@dataclass
class ProcessResult:
    audio: np.ndarray  # (channels, samples) float32
    sr: int
    notes: list[str] = field(default_factory=list)
    input_sr: int = 0
    detected_bandwidth_hz: float = 0.0


class Engine(Protocol):
    """An upscaling engine."""

    name: str

    def process(
        self,
        x: np.ndarray,
        sr: int,
        target_sr: int,
        preset: Preset,
        progress: ProgressCb | None = None,
    ) -> ProcessResult:
        ...
