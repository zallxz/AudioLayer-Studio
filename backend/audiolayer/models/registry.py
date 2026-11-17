"""Model catalogue: where weights come from, their licenses, and on-demand download.

Weights are never committed to the repository. They are downloaded on first use into
``AUDIOLAYER_MODEL_DIR`` (or fetched by the upstream library's own cache: Demucs uses the
torch hub cache, faster-whisper uses the Hugging Face cache). Set
``AUDIOLAYER_ALLOW_MODEL_DOWNLOAD=false`` to forbid downloads (air-gapped installs).
"""

from __future__ import annotations

import hashlib
import importlib.util
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import ModelUnavailable


@dataclass(frozen=True)
class ModelSpec:
    key: str
    role: str
    name: str
    license: str
    source: str
    size_mb: int
    python_deps: tuple[str, ...]
    url: str | None = None
    sha256: str | None = None
    filename: str | None = None


MODELS: dict[str, ModelSpec] = {
    "panns": ModelSpec(
        key="panns", role="sound event detection (527 AudioSet classes, framewise)",
        name="PANNs Cnn14_DecisionLevelMax", license="MIT (code & weights, Kong et al. 2020)",
        source="https://zenodo.org/record/3987831", size_mb=312, python_deps=("torch", "torchlibrosa"),
        url="https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1",
        sha256="dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142",
        filename="Cnn14_DecisionLevelMax_mAP=0.385.pth",
    ),
    "demucs": ModelSpec(
        key="demucs", role="source separation (vocals / drums / bass / other)",
        name="Demucs v4 htdemucs (Hybrid Transformer)", license="MIT (code & weights)",
        source="https://github.com/adefossez/demucs", size_mb=84, python_deps=("torch", "demucs"),
    ),
    "faster-whisper": ModelSpec(
        key="faster-whisper", role="speech recognition, timestamps, intelligibility",
        name="Whisper via faster-whisper (CTranslate2)", license="MIT (code & OpenAI Whisper weights)",
        source="https://github.com/SYSTRAN/faster-whisper", size_mb=145, python_deps=("faster_whisper",),
    ),
    "noisereduce": ModelSpec(
        key="noisereduce", role="stationary noise reduction (spectral gating)",
        name="noisereduce", license="MIT", source="https://github.com/timsainb/noisereduce",
        size_mb=0, python_deps=("noisereduce",),
    ),
}


def deps_available(spec: ModelSpec) -> bool:
    return all(importlib.util.find_spec(d) is not None for d in spec.python_deps)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_path(spec: ModelSpec, model_dir: Path) -> Path:
    assert spec.filename
    return model_dir / spec.key / spec.filename


def ensure_weights(spec: ModelSpec, model_dir: Path, *, allow_download: bool) -> Path:
    """Return the local weights path, downloading + verifying it if needed."""
    if not spec.url or not spec.filename:
        raise ValueError(f"{spec.key} manages its own weights")
    dst = weights_path(spec, model_dir)
    if dst.exists():
        return dst
    if not allow_download:
        raise ModelUnavailable(
            f"Weights for {spec.name} are not installed",
            hint=f"Run `python scripts/download_models.py {spec.key}` or place the file at {dst}",
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    try:
        with urllib.request.urlopen(spec.url, timeout=60) as r, tmp.open("wb") as f:  # noqa: S310 - fixed URL
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
        if spec.sha256 and _sha256(tmp) != spec.sha256:
            raise ModelUnavailable(f"Checksum mismatch for {spec.name}; download corrupted")
        tmp.replace(dst)
    except ModelUnavailable:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:  # network errors
        tmp.unlink(missing_ok=True)
        raise ModelUnavailable(f"Could not download {spec.name}: {e}", hint=f"Source: {spec.source}") from e
    return dst


def status(model_dir: Path) -> list[dict[str, object]]:
    out = []
    for spec in MODELS.values():
        installed: bool | None
        if spec.filename:
            installed = weights_path(spec, model_dir).exists()
        else:
            installed = None  # cached by the upstream library on first use
        out.append({
            "key": spec.key, "name": spec.name, "role": spec.role, "license": spec.license,
            "source": spec.source, "size_mb": spec.size_mb, "python_deps_installed": deps_available(spec),
            "weights_installed": installed,
        })
    return out
