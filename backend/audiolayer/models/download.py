"""Pre-download model weights (used by `audiolayer models download` and scripts/download_models.py)."""

from __future__ import annotations

from collections.abc import Callable

from ..config import Settings
from .registry import MODELS, ensure_weights

DEFAULT = ("panns", "demucs", "faster-whisper")
CLOUD = ("panns-onnx", "demucs-onnx", "faster-whisper")


def default_keys(settings: Settings) -> tuple[str, ...]:
    from ..core.components import pick_engine

    return CLOUD if pick_engine(settings) == "onnx" else DEFAULT


def download(settings: Settings, keys: tuple[str, ...] | list[str] = DEFAULT,
             log: Callable[[str], None] = print) -> None:
    for key in keys:
        if key not in MODELS:
            raise ValueError(f"unknown model {key!r}; choose from {list(MODELS)}")
        spec = MODELS[key]
        log(f"→ {spec.name} [{spec.license}] (~{spec.size_mb} MB)")
        if key in ("panns", "panns-onnx", "demucs-onnx", "whisper-base-mirror"):
            log(f"   {ensure_weights(spec, settings.model_dir, allow_download=True)}")
        elif key == "demucs":
            from demucs.pretrained import get_model

            get_model(settings.demucs_model)
            log("   cached in the torch hub directory")
        elif key == "faster-whisper":
            from ..classification.speech import WhisperAsr

            WhisperAsr(settings)._model()  # Hugging Face first, GitHub mirror as fallback
            log(f"   whisper-{settings.whisper_model} ready")
        elif key == "noisereduce":
            log("   no weights needed")
