"""Resolve configured model adapters (with graceful fallbacks) and report capabilities."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, resolve_device
from ..models.registry import MODELS, deps_available, status


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@dataclass
class Components:
    classifier: Any
    separator: Any | None
    asr: Any | None
    notes: list[str] = field(default_factory=list)


def build_components(settings: Settings) -> Components:
    notes: list[str] = []
    # classifier
    clf: Any
    want = settings.classifier
    if want in ("panns", "auto") and deps_available(MODELS["panns"]):
        from ..classification.panns import PannsClassifier

        clf = PannsClassifier(settings)
    else:
        if want == "panns":
            notes.append("PANNs requested but torch/torchlibrosa are not installed; using heuristic classifier.")
        elif want == "auto":
            notes.append("ML extras not installed: using the DSP heuristic classifier (low accuracy).")
        from ..classification.heuristic import HeuristicClassifier

        clf = HeuristicClassifier()
    # separator
    sep: Any | None = None
    if settings.separator in ("demucs", "auto") and deps_available(MODELS["demucs"]):
        from ..separation.demucs_sep import DemucsSeparator

        sep = DemucsSeparator(settings)
    elif settings.separator != "none":
        notes.append("No source-separation model available (install the `ml` extra for Demucs). "
                     "Detection works; isolation/removal of categories is disabled.")
    # asr
    asr: Any | None = None
    if settings.asr in ("faster-whisper", "auto") and deps_available(MODELS["faster-whisper"]):
        from ..classification.speech import WhisperAsr

        asr = WhisperAsr(settings)
    elif settings.asr != "none":
        notes.append("faster-whisper not installed: no transcripts / SRT / gibberish detection.")
    return Components(classifier=clf, separator=sep, asr=asr, notes=notes)


def system_info(settings: Settings, comps: Components) -> dict[str, Any]:
    return {
        "device": resolve_device(settings.device),
        "classifier": getattr(comps.classifier, "name", None),
        "classifier_calibrated": getattr(comps.classifier, "calibrated", False),
        "separator": getattr(comps.separator, "name", None),
        "asr": getattr(comps.asr, "name", None),
        "denoiser": "noisereduce + scipy filters" if _has("noisereduce") else "scipy filters only",
        "notes": comps.notes,
        "models": status(settings.model_dir),
        "limits": {"max_upload_mb": settings.max_upload_mb, "max_duration_s": settings.max_duration_s},
        "auth_enabled": settings.auth_enabled,
    }
