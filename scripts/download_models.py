#!/usr/bin/env python3
"""Download / pre-cache model weights (optional — they are also fetched on first use).

    python scripts/download_models.py            # all models
    python scripts/download_models.py panns demucs faster-whisper
    python scripts/download_models.py --list     # show status, sizes and licenses

Weights go to AUDIOLAYER_MODEL_DIR (PANNs, Whisper) or the torch hub cache (Demucs).
Nothing is ever written into the repository.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    from audiolayer.config import get_settings
    from audiolayer.models.registry import MODELS, ensure_weights, status

    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["panns", "demucs", "faster-whisper"])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    if a.list:
        print(json.dumps(status(s.model_dir), indent=2))
        return 0
    for key in a.models:
        if key not in MODELS:
            print(f"unknown model {key!r}; choose from {list(MODELS)}", file=sys.stderr)
            return 2
        spec = MODELS[key]
        print(f"→ {spec.name} [{spec.license}] (~{spec.size_mb} MB)")
        if key == "panns":
            print("  ", ensure_weights(spec, s.model_dir, allow_download=True))
        elif key == "demucs":
            from demucs.pretrained import get_model

            get_model(s.demucs_model)
            print("   cached in the torch hub directory")
        elif key == "faster-whisper":
            from faster_whisper import WhisperModel

            WhisperModel(s.whisper_model, device="cpu", compute_type="int8", download_root=str(s.model_dir / "whisper"))
            print(f"   whisper-{s.whisper_model} cached in {s.model_dir / 'whisper'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
