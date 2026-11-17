"""CLI: Claude Desktop config merge and argument parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audiolayer import cli


def test_setup_claude_merges_and_backs_up(tmp_path: Path, capsys):
    cfg = tmp_path / "claude_desktop_config.json"
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "theme": "dark"}))
    inputs = tmp_path / "clips"
    rc = cli.main(["setup-claude", "--input-dir", str(inputs), "--config", str(cfg)])
    assert rc == 0
    data = json.loads(cfg.read_text())
    assert data["theme"] == "dark" and "other" in data["mcpServers"]
    entry = data["mcpServers"]["audiolayer"]
    assert entry["args"] == ["-m", "audiolayer.mcp_server.server"]
    assert entry["env"]["AUDIOLAYER_INPUT_DIRS"] == str(inputs.resolve())
    assert inputs.is_dir()
    assert list(tmp_path.glob("claude_desktop_config.json.bak-*"))


def test_setup_claude_dry_run_and_invalid_json(tmp_path: Path, capsys):
    cfg = tmp_path / "c.json"
    cli.main(["setup-claude", "--input-dir", str(tmp_path), "--config", str(cfg), "--dry-run"])
    assert not cfg.exists()
    assert '"audiolayer"' in capsys.readouterr().out
    cfg.write_text("{broken")
    with pytest.raises(SystemExit):
        cli.main(["setup-claude", "--input-dir", str(tmp_path), "--config", str(cfg)])


def test_clean_rejects_unknown_category(tmp_path: Path):
    with pytest.raises(SystemExit):
        cli.main(["clean", str(tmp_path / "x.mp4"), "--remove", "drums"])


def test_parser_has_all_commands():
    p = cli.build_parser()
    for cmd in (["doctor"], ["web", "--no-browser"], ["models", "list"], ["mcp"], ["analyze", "a.mp4"],
                ["clean", "a.mp4", "--preset", "remove_music", "--loudness", "youtube", "--video"],
                ["clean", "--project", "prj_abc", "--remove", "bgm,music"]):
        assert p.parse_args(cmd).fn
