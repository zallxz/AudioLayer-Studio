"""Speech recognition + intelligibility scoring with faster-whisper (optional).

Besides transcripts and SRT, the ASR confidence is used to flag *unintelligible* speech:
AI video generators frequently produce gibberish "dialogue" (tonal babble, half-words,
fake languages). Whisper's average log-probability, no-speech probability, compression
ratio and language-ID confidence give a usable signal for that. The flag is a hint, not a
verdict — real mumbled or accented speech can also score low.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..audio import io as aio
from ..config import Settings, resolve_device
from ..core.schemas import Segment
from .base import CancelFn, ProgressFn

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float

    @property
    def intelligibility(self) -> float:
        return float(math.exp(min(0.0, self.avg_logprob)) * (1 - self.no_speech_prob))


@dataclass
class AsrResult:
    language: str | None
    language_probability: float
    segments: list[AsrSegment]
    model: str


class WhisperAsr:
    name = "faster-whisper"
    license = "MIT"

    def __init__(self, settings: Settings):
        self.settings = settings
        dev = resolve_device(settings.device)
        self.device = "cuda" if dev == "cuda" else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

    def _model(self) -> Any:
        from faster_whisper import WhisperModel

        key = f"{self.settings.whisper_model}:{self.device}"
        with _LOCK:
            if key not in _CACHE:
                _CACHE[key] = WhisperModel(
                    self.settings.whisper_model, device=self.device, compute_type=self.compute_type,
                    download_root=str(self.settings.model_dir / "whisper"),
                    local_files_only=not self.settings.allow_model_download,
                )
            return _CACHE[key]

    def transcribe(self, path: Path, language: str | None = None, on_progress: ProgressFn | None = None,
                   check_cancel: CancelFn | None = None) -> AsrResult:
        sr, _, n = aio.info(path)
        # Whisper wants 16 kHz mono; decode in blocks to bound memory.
        parts = [aio.resample(aio.to_mono(b), sr, 16000) for _, b in aio.iter_blocks(path, sr * 60)]
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        model = self._model()
        segs_iter, info = model.transcribe(
            audio, language=language, vad_filter=True, beam_size=1, condition_on_previous_text=False,
        )
        out: list[AsrSegment] = []
        total = max(1e-6, n / sr)
        for s in segs_iter:
            if check_cancel:
                check_cancel()
            out.append(AsrSegment(
                start=float(s.start), end=float(s.end), text=s.text.strip(),
                avg_logprob=float(s.avg_logprob), no_speech_prob=float(s.no_speech_prob),
                compression_ratio=float(s.compression_ratio),
            ))
            if on_progress:
                on_progress(min(1.0, s.end / total))
        return AsrResult(language=info.language, language_probability=float(info.language_probability),
                         segments=out, model=f"whisper-{self.settings.whisper_model}")


def attach_transcripts(dialogue: list[Segment], asr: AsrResult, duration: float) -> list[Segment]:
    """Attach text + intelligibility to dialogue segments; add segments ASR found but SED missed."""
    used: set[int] = set()
    for seg in dialogue:
        hits = [(i, a) for i, a in enumerate(asr.segments) if a.end > seg.start and a.start < seg.end]
        if hits:
            used.update(i for i, _ in hits)
            seg.transcript = " ".join(a.text for _, a in hits).strip() or None
            seg.intelligibility = round(float(np.mean([a.intelligibility for _, a in hits])), 3)
            seg.language = asr.language
            if seg.intelligibility < 0.35:
                seg.flags.append("unintelligible")
            if any(a.compression_ratio > 2.4 for _, a in hits):
                seg.flags.append("repetitive_or_hallucinated")
        elif seg.confidence > 0.5 and seg.duration > 0.6:
            seg.flags.append("no_words_recognized")
        if asr.language_probability < 0.5 and seg.transcript:
            seg.flags.append("language_uncertain")
    added: list[Segment] = []
    for i, a in enumerate(asr.segments):
        if i in used or a.no_speech_prob > 0.6 or not a.text:
            continue
        added.append(Segment(
            category="dialogue", start=round(a.start, 3), end=round(min(a.end, duration), 3),
            confidence=round(min(0.9, a.intelligibility + 0.2), 3), uncertain=a.intelligibility < 0.45,
            stream="voice", description="Speech (found by ASR)", transcript=a.text,
            intelligibility=round(a.intelligibility, 3), language=asr.language,
            flags=["asr_only"] + (["unintelligible"] if a.intelligibility < 0.35 else []),
        ))
    return dialogue + added
