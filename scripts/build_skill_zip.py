"""Build skills/audiolayer.zip (the file users upload in Claude → Settings → Capabilities → Skills).

Run after editing skills/audiolayer/SKILL.md:  python scripts/build_skill_zip.py
tests/test_skill_zip.py fails if the ZIP is out of date.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "skills" / "audiolayer"
OUT = ROOT / "skills" / "audiolayer.zip"


def build(out: Path = OUT) -> Path:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(SRC.rglob("*")):
            if f.is_file():
                info = zipfile.ZipInfo(str(Path("audiolayer") / f.relative_to(SRC)), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, f.read_bytes())
    return out


if __name__ == "__main__":
    print(build())
