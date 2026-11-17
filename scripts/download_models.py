#!/usr/bin/env python3
"""Download / pre-cache model weights. Same as `audiolayer models download`.

    python scripts/download_models.py            # all models
    python scripts/download_models.py panns demucs faster-whisper
    python scripts/download_models.py --list     # show status, sizes and licenses
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    from audiolayer.cli import main

    args = sys.argv[1:]
    raise SystemExit(main(["models", "list"] if args == ["--list"] else ["models", "download", *args]))
