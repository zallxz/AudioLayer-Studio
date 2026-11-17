"""Pre-download model weights (used by `audiolayer models download` and scripts/download_models.py)."""

from __future__ import annotations

from collections.abc import Callable

from ..config import Settings
from .registry import MODELS, ensure_weights

DEFAULT = ("panns", "demucs", "faster-whisper")


def download(settings: Settings, keys: tuple[str, ...] | list[str] = DEFAULT,
             log: Callable[[str], None] = print) -> None:
    for key in keys:
        if key not in MODELS:
            raise ValueError(f"unknown model {key!r}; choose from {list(MODELS)}")
        spec = MODELS[key]
        log(f"→ {spec.name} [{spec.license}] (~{spec.size_mb} MB)")
        if key == "panns":
            log(f"   {ensure_weights(spec, settings.model_dir, allow_download=True)}")
        elif key == "demucs":
            from demucs.pretrained import get_model

            get_model(settings.demucs_model)
            log("   cached in the torch hub directory")
        elif key == "faster-whisper":
            from faster_whisper import WhisperModel

            WhisperModel(settings.whisper_model, device="cpu", compute_type="int8",
                         download_root=str(settings.model_dir / "whisper"))
            log(f"   whisper-{settings.whisper_model} cached in {settings.model_dir / 'whisper'}")
        elif key == "noisereduce":
            log("   no weights needed")
