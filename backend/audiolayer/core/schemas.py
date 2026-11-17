"""Pydantic data model shared by the pipeline, REST API and MCP server."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

Category = Literal["dialogue", "bgm", "sfx", "music", "noise"]
CATEGORIES: tuple[Category, ...] = ("dialogue", "bgm", "sfx", "music", "noise")

# A "stream" is a separated signal produced by the separation stage. Segments live on
# streams; categories are labels routed from streams. Changing a label never requires
# re-running separation.
Stream = Literal["voice", "music", "fx", "noise", "mix"]
STREAMS: tuple[Stream, ...] = ("voice", "music", "fx", "noise")
DEFAULT_STREAM_CATEGORY: dict[str, Category] = {
    "voice": "dialogue",
    "music": "bgm",
    "fx": "sfx",
    "noise": "noise",
}
CATEGORY_DEFAULT_STREAM: dict[str, Stream] = {
    "dialogue": "voice",
    "bgm": "music",
    "music": "music",
    "sfx": "fx",
    "noise": "noise",
}

CATEGORY_DEFINITIONS: dict[str, str] = {
    "dialogue": "Human speech and other spoken/vocal content: conversation, narration, voice-over, "
    "whispering, shouting, laughter, breaths.",
    "bgm": "Background music: score, ambient/cinematic beds and instrumental tracks that sit *under* "
    "dialogue or action.",
    "sfx": "Sound effects and intentional ambience: whooshes, impacts, footsteps, foley, mechanical, "
    "UI and environmental sounds (rain, wind, crowd walla).",
    "music": "Foreground music: songs, singing and performances that are the focus rather than a bed. "
    "BGM vs Music is a *role* decision; the audio alone often cannot tell — correct it manually.",
    "noise": "Unwanted technical contamination: hiss, hum, static, mic wind noise, rumble, clipping, "
    "codec artifacts. Intentional ambience is NOT noise.",
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class LabelScore(BaseModel):
    label: str
    score: float


class Segment(BaseModel):
    id: str = Field(default_factory=lambda: new_id("seg"))
    category: Category
    start: float
    end: float
    confidence: float = Field(ge=0.0, le=1.0)
    uncertain: bool = False
    stream: Stream = "mix"
    description: str = ""
    labels: list[LabelScore] = Field(default_factory=list)
    transcript: str | None = None
    intelligibility: float | None = None
    language: str | None = None
    flags: list[str] = Field(default_factory=list)
    muted: bool = False
    solo: bool = False
    edited: bool = False
    original_category: Category | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        return round(max(0.0, self.end - self.start), 4)


class MediaInfo(BaseModel):
    filename: str
    container: str = ""
    size_bytes: int = 0
    has_video: bool = False
    has_audio: bool = True
    video_duration: float | None = None
    audio_duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    audio_codec: str | None = None
    video_codec: str | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    audio_start_offset: float = 0.0
    working_sample_rate: int = 48000
    working_channels: int = 2


class QualityIssue(BaseModel):
    id: str = Field(default_factory=lambda: new_id("iss"))
    kind: str
    severity: Literal["info", "warning", "error"] = "warning"
    title: str
    detail: str
    start: float | None = None
    end: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    fix: str | None = None  # id of a preset/action that addresses it


class Loudness(BaseModel):
    integrated_lufs: float | None = None
    true_peak_dbtp: float | None = None
    lra: float | None = None


class ModelUse(BaseModel):
    role: str
    name: str
    license: str
    note: str = ""


class Analysis(BaseModel):
    created_at: float = Field(default_factory=time.time)
    classifier: str
    separator: str
    asr: str
    frame_hop: float
    separated: bool
    loudness: Loudness = Field(default_factory=Loudness)
    issues: list[QualityIssue] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    models: list[ModelUse] = Field(default_factory=list)
    category_presence: dict[str, float] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)


class MixState(BaseModel):
    removed: list[Category] = Field(default_factory=list)


class Project(BaseModel):
    id: str = Field(default_factory=lambda: new_id("prj"))
    name: str
    owner: str = "local"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    status: Literal["uploaded", "analyzing", "analyzed", "error"] = "uploaded"
    error: str | None = None
    media: MediaInfo
    original_file: str  # relative to project dir
    analysis: Analysis | None = None
    segments: list[Segment] = Field(default_factory=list)
    mix: MixState = Field(default_factory=MixState)
    history: list[dict[str, Any]] = Field(default_factory=list)
    redo: list[dict[str, Any]] = Field(default_factory=list)
    stems_version: int = 0
    exports: list[dict[str, Any]] = Field(default_factory=list)


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class Job(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    project_id: str
    kind: str
    owner: str = "local"
    status: JobStatus = "queued"
    progress: float = 0.0
    stage: str = "queued"
    message: str = ""
    error: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancellable: bool = True


class AnalyzeOptions(BaseModel):
    separate: bool = Field(True, description="Run source separation (needed for removal/isolation).")
    transcribe: bool = Field(True, description="Transcribe dialogue when an ASR model is available.")
    denoise: Literal["auto", "on", "off"] = Field(
        "auto", description="Build the noise stem: auto = only when hiss/hum is detected."
    )
    language: str | None = Field(None, description="ISO language hint for transcription.")


ExportItem = Literal[
    "mix", "dialogue", "bgm", "sfx", "music", "noise", "original", "video", "metadata"
]

DEFAULT_EXPORT_ITEMS: tuple[ExportItem, ...] = ("mix", "dialogue", "bgm", "sfx", "music", "noise", "original", "metadata")


class ExportOptions(BaseModel):
    items: list[ExportItem] = Field(
        default_factory=lambda: list(DEFAULT_EXPORT_ITEMS)
    )
    format: Literal["wav", "flac", "mp3"] = "wav"
    remove: list[Category] | None = Field(
        None, description="Categories removed from the cleaned mix. None = current project selection."
    )
    loudness_preset: Literal["none", "youtube", "tiktok", "podcast", "broadcast"] = "none"
    edge_fade_ms: float = Field(5.0, ge=0, le=500)
    zip: bool = True
