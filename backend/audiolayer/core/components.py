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
    engine: str = "custom"


def pick_engine(settings: Settings) -> str:
    torch_ok = deps_available(MODELS["demucs"]) and deps_available(MODELS["panns"])
    onnx_ok = _has("onnxruntime")
    if settings.engine == "torch" or (settings.engine == "auto" and torch_ok):
        return "torch" if torch_ok else "none"
    if settings.engine in ("onnx", "auto") and onnx_ok:
        return "onnx"
    return "none"


def build_components(settings: Settings) -> Components:
    notes: list[str] = []
    engine = pick_engine(settings)
    if settings.engine == "torch" and engine == "none":
        notes.append("AUDIOLAYER_ENGINE=torch but PyTorch/Demucs are not installed.")
    # classifier
    clf: Any
    if settings.classifier == "heuristic" or engine == "none":
        if settings.classifier != "heuristic":
            notes.append("No ML engine installed (install the `ml` or `cloud` extra): using the DSP heuristic "
                         "classifier (low accuracy).")
        from ..classification.heuristic import HeuristicClassifier

        clf = HeuristicClassifier()
    elif engine == "torch":
        from ..classification.panns import PannsClassifier

        clf = PannsClassifier(settings)
    else:
        from ..classification.panns import PannsOnnxClassifier

        clf = PannsOnnxClassifier(settings)
    # separator
    sep: Any | None = None
    if settings.separator == "none":
        pass
    elif engine == "torch":
        from ..separation.demucs_sep import DemucsSeparator

        sep = DemucsSeparator(settings)
    elif engine == "onnx":
        from ..separation.demucs_onnx import DemucsOnnxSeparator

        sep = DemucsOnnxSeparator(settings)
    else:
        notes.append("No source-separation model available (install the `ml` or `cloud` extra). "
                     "Detection works; isolation/removal of categories is disabled.")
    # asr
    asr: Any | None = None
    if settings.asr in ("faster-whisper", "auto") and deps_available(MODELS["faster-whisper"]):
        from ..classification.speech import WhisperAsr

        asr = WhisperAsr(settings)
    elif settings.asr != "none":
        notes.append("faster-whisper not installed: no transcripts / SRT / gibberish detection.")
    return Components(classifier=clf, separator=sep, asr=asr, notes=notes, engine=engine)


def system_info(settings: Settings, comps: Components) -> dict[str, Any]:
    return {
        "device": resolve_device(settings.device) if comps.engine == "torch" else "cpu",
        "engine": comps.engine,
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
