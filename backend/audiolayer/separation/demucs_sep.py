"""Demucs v4 (htdemucs) neural source separation adapter."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..audio import io as aio
from ..config import Settings, resolve_device
from ..core.errors import ModelUnavailable
from ..models.registry import MODELS
from .base import CancelFn, ProgressFn

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


class DemucsSeparator:
    license = MODELS["demucs"].license

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.name = f"demucs-{settings.demucs_model}"

    def _model(self) -> Any:
        with _LOCK:
            key = f"{self.settings.demucs_model}:{self.device}"
            if key not in _CACHE:
                import torch
                from demucs.pretrained import get_model

                if not self.settings.allow_model_download:
                    hub = Path(torch.hub.get_dir()) / "checkpoints"
                    if not hub.exists() or not any(hub.glob("*.th")):
                        raise ModelUnavailable(
                            "Demucs weights are not cached and downloads are disabled",
                            hint="Run scripts/download_models.py demucs",
                        )
                model = get_model(self.settings.demucs_model)
                model.eval().to(self.device)
                _CACHE[key] = model
            return _CACHE[key]

    def separate(self, master: Path, out_dir: Path, on_progress: ProgressFn | None = None,
                 check_cancel: CancelFn | None = None) -> dict[str, Path]:
        import torch
        from demucs.apply import apply_model

        model = self._model()
        msr = int(model.samplerate)
        vocal_idx = model.sources.index("vocals")
        sr, ch, _ = aio.info(master)

        def fn(x: np.ndarray) -> dict[str, np.ndarray]:
            stereo = x if x.shape[1] == 2 else np.repeat(x[:, :1], 2, axis=1)
            y = aio.resample(stereo, sr, msr)
            wav = torch.from_numpy(np.ascontiguousarray(y.T))[None]
            ref = wav.mean(0).mean(0)
            mean, std = ref.mean(), ref.std() + 1e-8
            with torch.no_grad():
                est = apply_model(model, (wav - mean) / std, device=self.device, split=True,
                                  overlap=0.25, progress=False)[0]
            est = est * std + mean
            voc = est[vocal_idx].cpu().numpy().T
            voc = aio.fit_length(aio.resample(voc, msr, sr), len(x))
            if ch == 1:
                voc = voc.mean(axis=1, keepdims=True)
            # accompaniment is the exact complement so the two always sum to the input
            return {"vocals": voc, "accompaniment": x - voc}

        out = {k: out_dir / f"{k}.wav" for k in ("vocals", "accompaniment")}
        aio.process_chunked(master, ["vocals", "accompaniment"], out, fn,
                            chunk_s=self.settings.chunk_seconds, overlap_s=2.0,
                            on_progress=on_progress, check_cancel=check_cancel)
        return out
