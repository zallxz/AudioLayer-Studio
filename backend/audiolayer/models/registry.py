"""Model catalogue: where weights come from, their licenses, and on-demand download.

Weights are never committed to the repository. They are downloaded on first use into
``AUDIOLAYER_MODEL_DIR`` (or fetched by the upstream library's own cache: Demucs uses the
torch hub cache, faster-whisper uses the Hugging Face cache). Set
``AUDIOLAYER_ALLOW_MODEL_DOWNLOAD=false`` to forbid downloads (air-gapped installs).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import ModelUnavailable

# Weights mirrored on GitHub (branch `model-weights`, files split into <=45 MB parts) so the torch-free
# engine works in sandboxes that can reach GitHub/PyPI but not Hugging Face or Zenodo.
MIRROR = os.environ.get(
    "AUDIOLAYER_MODEL_MIRROR", "https://raw.githubusercontent.com/zallxz/AudioLayer-Studio/model-weights"
).rstrip("/")


@dataclass(frozen=True)
class MirrorFile:
    path: str  # path inside the mirror (and inside the local model folder)
    sha256: str
    parts: int = 1


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
    mirror: tuple[MirrorFile, ...] = field(default_factory=tuple)


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
    # ---- torch-free "cloud" engine (ONNX Runtime); weights mirrored on GitHub -------------------
    "panns-onnx": ModelSpec(
        key="panns-onnx", role="sound event detection (ONNX)",
        name="PANNs Cnn14_DecisionLevelMax (ONNX, fp16 weights)", license="MIT (Kong et al. 2020)",
        source="exported by scripts/export_onnx.py", size_mb=164, python_deps=("onnxruntime",),
        filename="panns_cnn14_sed.onnx",
        mirror=(MirrorFile("panns_cnn14_sed.onnx",
                           "fa14ee879a128a80faa38bc10c5f0f3ef0fe025599cdfc642cbc3443f8ce34ec", 4),),
    ),
    "demucs-onnx": ModelSpec(
        key="demucs-onnx", role="source separation (ONNX)",
        name="Demucs v4 htdemucs core (ONNX, fp16 weights)", license="MIT (Meta / Défossez)",
        source="exported by scripts/export_onnx.py", size_mb=91, python_deps=("onnxruntime",),
        filename="htdemucs_core.onnx",
        mirror=(MirrorFile("htdemucs_core.onnx",
                           "65020db53f86ba8a453074ce800f856b3fbaccf056c717a7d9e73b327ed3ad11", 2),),
    ),
    "whisper-base-mirror": ModelSpec(
        key="whisper-base-mirror", role="speech recognition weights (GitHub mirror)",
        name="Whisper base (CTranslate2, SYSTRAN conversion)", license="MIT",
        source="https://huggingface.co/Systran/faster-whisper-base", size_mb=148,
        python_deps=("faster_whisper",), filename="whisper-base/model.bin",
        mirror=(
            MirrorFile("whisper-base/model.bin", "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9", 4),
            MirrorFile("whisper-base/config.json", "56a6d8110d311f19c8f0471e562832c7527f146b567275bfca59fcf7c184da9a"),
            MirrorFile("whisper-base/tokenizer.json", "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
            MirrorFile("whisper-base/vocabulary.txt", "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
        ),
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


def _fetch(url: str, f: object) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "audiolayer-studio"})  # noqa: S310 - fixed https URLs
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)  # type: ignore[attr-defined]


def _download_mirror(spec: ModelSpec, folder: Path) -> None:
    for mf in spec.mirror:
        dst = folder / mf.path
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".download")
        with tmp.open("wb") as f:
            if mf.parts == 1:
                _fetch(f"{MIRROR}/{mf.path}", f)
            else:
                for i in range(mf.parts):
                    _fetch(f"{MIRROR}/{mf.path}.part{i:02d}", f)
        if _sha256(tmp) != mf.sha256:
            tmp.unlink(missing_ok=True)
            raise ModelUnavailable(f"Checksum mismatch for {mf.path}; download corrupted")
        tmp.replace(dst)


def ensure_weights(spec: ModelSpec, model_dir: Path, *, allow_download: bool) -> Path:
    """Return the local weights path, downloading + verifying it if needed.

    Tries the original source first (when the model has one), then the GitHub mirror.
    """
    if not spec.filename or not (spec.url or spec.mirror):
        raise ValueError(f"{spec.key} manages its own weights")
    dst = weights_path(spec, model_dir)
    if dst.exists():
        return dst
    if not allow_download:
        raise ModelUnavailable(
            f"Weights for {spec.name} are not installed",
            hint=f"Run `audiolayer models download {spec.key}` or place the file at {dst}",
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    if spec.url:
        tmp = dst.with_suffix(".part")
        try:
            with tmp.open("wb") as f:
                _fetch(spec.url, f)
            if spec.sha256 and _sha256(tmp) != spec.sha256:
                raise ModelUnavailable(f"Checksum mismatch for {spec.name}; download corrupted")
            tmp.replace(dst)
            return dst
        except Exception as e:  # noqa: BLE001 - fall through to the mirror
            tmp.unlink(missing_ok=True)
            errors.append(f"{spec.url}: {e}")
    if spec.mirror:
        try:
            _download_mirror(spec, model_dir / spec.key)
            return dst
        except Exception as e:  # noqa: BLE001
            errors.append(f"{MIRROR}: {e}")
    raise ModelUnavailable(f"Could not download {spec.name}: " + " | ".join(errors), hint=f"Source: {spec.source}")


def status(model_dir: Path) -> list[dict[str, object]]:
    out = []
    for spec in MODELS.values():
        installed: bool | None
        if spec.filename and (spec.url or spec.mirror):
            installed = weights_path(spec, model_dir).exists()
        else:
            installed = None  # cached by the upstream library on first use
        out.append({
            "key": spec.key, "name": spec.name, "role": spec.role, "license": spec.license,
            "source": spec.source, "size_mb": spec.size_mb, "python_deps_installed": deps_available(spec),
            "weights_installed": installed,
        })
    return out
