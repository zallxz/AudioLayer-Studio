"""PANNs CNN14 sound-event-detection adapter (framewise, multi-label → supports overlap)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..audio import io as aio
from ..config import Settings, resolve_device
from ..models.registry import MODELS, ensure_weights
from .base import CancelFn, FrameProbs, ProgressFn, audioset_labels, audioset_to_tracks

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, Any] = {}


class PannsClassifier:
    name = "panns-cnn14-sed"
    license = MODELS["panns"].license
    calibrated = True

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = resolve_device(settings.device)
        if self.device == "mps":
            self.device = "cpu"  # some ops in torchlibrosa are not MPS-friendly

    def _model(self) -> Any:
        import torch

        from ..models.panns_cnn14 import Cnn14DecisionLevelMax

        with _MODEL_LOCK:
            key = f"panns:{self.device}"
            if key not in _MODEL_CACHE:
                w = ensure_weights(MODELS["panns"], self.settings.model_dir,
                                   allow_download=self.settings.allow_model_download)
                model = Cnn14DecisionLevelMax()
                ckpt = torch.load(str(w), map_location="cpu", weights_only=True)
                state = ckpt.get("model", ckpt)
                model.load_state_dict(state, strict=False)
                model.eval().to(self.device)
                _MODEL_CACHE[key] = model
            return _MODEL_CACHE[key]

    def _run(self, y: np.ndarray) -> np.ndarray:
        """(samples,) at 32 kHz -> (frames, 527) probabilities."""
        import torch

        model = self._model()
        with torch.no_grad():
            return np.asarray(model(torch.from_numpy(y[None, :]).to(self.device))[0].cpu().numpy())

    def frame_probs(self, path: Path, on_progress: ProgressFn | None = None,
                    check_cancel: CancelFn | None = None) -> FrameProbs:
        HOP, POOL_RATIO, SAMPLE_RATE = 320, 32, 32000
        FRAME_HOP_S = HOP * POOL_RATIO / SAMPLE_RATE
        sr, _, n = aio.info(path)
        unit = HOP * POOL_RATIO  # samples per output frame at 32 kHz
        frames_per_chunk = max(1, int(self.settings.chunk_seconds / FRAME_HOP_S))
        chunk_in = int(round(frames_per_chunk * FRAME_HOP_S * sr))
        duration = n / sr
        total_frames = int(np.ceil(duration / FRAME_HOP_S))
        outs: list[np.ndarray] = []
        labels = audioset_labels()
        pos = 0
        while pos < n:
            if check_cancel:
                check_cancel()
            x = aio.read_range(path, pos, pos + chunk_in)
            mono = aio.to_mono(x)
            y = aio.resample(mono, sr, SAMPLE_RATE)
            want = int(np.ceil(len(y) / unit)) * unit
            y = np.pad(y, (0, want - len(y))).astype(np.float32)
            p = self._run(y)
            outs.append(p[: want // unit])
            pos += chunk_in
            if on_progress:
                on_progress(min(1.0, pos / max(n, 1)))
        allp = np.concatenate(outs, axis=0)[:total_frames] if outs else np.zeros((0, len(labels)))
        tracks = audioset_to_tracks(allp)
        top: list[list[tuple[str, float]]] = []
        for row in allp:
            idx = np.argsort(row)[-4:][::-1]
            top.append([(labels[i], round(float(row[i]), 3)) for i in idx if row[i] > 0.05])
        return FrameProbs(hop=FRAME_HOP_S, probs=tracks, top_labels=top)


class PannsOnnxClassifier(PannsClassifier):
    """Same model through ONNX Runtime — no PyTorch needed (torch-free "cloud" engine)."""

    name = "panns-cnn14-sed-onnx"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = "cpu"

    def _session(self) -> Any:
        import onnxruntime as ort

        with _MODEL_LOCK:
            if "panns-onnx" not in _MODEL_CACHE:
                w = ensure_weights(MODELS["panns-onnx"], self.settings.model_dir,
                                   allow_download=self.settings.allow_model_download)
                _MODEL_CACHE["panns-onnx"] = ort.InferenceSession(str(w), providers=["CPUExecutionProvider"])
            return _MODEL_CACHE["panns-onnx"]

    def _run(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(self._session().run(None, {"waveform": y[None, :].astype(np.float32)})[0][0])
