import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_skill_zip_matches_skill_folder():
    zpath = ROOT / "skills" / "audiolayer.zip"
    assert zpath.exists(), "run: python scripts/build_skill_zip.py"
    with zipfile.ZipFile(zpath) as zf:
        assert zf.read("audiolayer/SKILL.md") == (ROOT / "skills/audiolayer/SKILL.md").read_bytes(), \
            "skills/audiolayer.zip is stale — run: python scripts/build_skill_zip.py"
