"""`audiolayer` — one command for everything.

    audiolayer doctor                        check FFmpeg, PyTorch, models, Claude setup
    audiolayer models download               fetch model weights (~550 MB, once)
    audiolayer web                           start the app and open it in the browser
    audiolayer clean clip.mp4 --preset remove_music --loudness youtube --video
    audiolayer analyze clip.mp4              report what is in the audio
    audiolayer setup-claude --input-dir ~/Videos   connect to Claude Desktop (and Claude Code)
    audiolayer mcp                           run the MCP server on stdio (used by Claude)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from . import __version__
from .config import get_settings

CATS = ("dialogue", "bgm", "sfx", "music", "noise")
PRESET_HELP = ("remove_music, dialogue_only, music_and_effects, remove_noise, remove_sfx, "
               "remove_gibberish, clear")


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _bar(label: str, frac: float, msg: str = "") -> None:
    width = 28
    n = int(frac * width)
    _err_line = f"\r{label} [{'#' * n}{'.' * (width - n)}] {frac * 100:5.1f}%  {msg[:48]:<48}"
    sys.stderr.write(_err_line)
    sys.stderr.flush()


def _wait(svc: Any, job_id: str, label: str, quiet: bool) -> Any:
    tty = sys.stderr.isatty()
    last = ""
    while True:
        j = svc.job(job_id, "local")
        msg = j.message or j.stage
        if not quiet:
            if tty:
                _bar(label, j.progress, msg)
            elif msg != last:  # logs / Claude Code: one line per stage instead of a redrawn bar
                _err(f"{label}: {msg} ({j.progress * 100:.0f}%)")
                last = msg
        if j.status not in ("queued", "running"):
            if not quiet and tty:
                sys.stderr.write("\n")
            return j
        time.sleep(0.5)


# ---------------------------------------------------------------------------- doctor
def cmd_doctor(_: argparse.Namespace) -> int:
    from .models.registry import status

    s = get_settings()
    ok = True

    def line(good: bool | None, name: str, detail: str) -> None:
        mark = "OK " if good else ("-- " if good is None else "!! ")
        print(f"  [{mark}] {name:<22} {detail}")

    print(f"AudioLayer Studio {__version__} · Python {platform.python_version()} · {platform.system()}")
    py_ok = sys.version_info >= (3, 11)
    line(py_ok, "Python >= 3.11", sys.executable)
    ok &= py_ok
    from .audio.ffmpeg import ffmpeg_bin

    try:
        ffp: str | None = ffmpeg_bin()
    except Exception:  # noqa: BLE001
        ffp = None
    line(bool(ffp), "ffmpeg", ffp or "not found — install FFmpeg (docs/TUTORIAL.en.md step 2) or `pip install imageio-ffmpeg`")
    ok &= bool(ffp)
    probe = shutil.which("ffprobe")
    av_ok = importlib.util.find_spec("av") is not None
    line(bool(probe) or av_ok, "ffprobe", probe or ("using PyAV instead" if av_ok else "not found"))
    ok &= bool(probe) or av_ok
    from .core.components import pick_engine

    engine = pick_engine(s)
    detail = {"torch": "PyTorch (full)", "onnx": "ONNX Runtime (torch-free cloud engine)",
              "none": "none → detection-only heuristic (pip install -e '.[ml]' or '.[cloud]')"}[engine]
    if engine == "torch":
        from .config import resolve_device

        detail += f", device: {resolve_device(s.device)}"
    line(engine != "none", "AI engine", detail)
    ok &= engine != "none"
    relevant = {"torch": {"panns", "demucs", "faster-whisper", "noisereduce"},
                "onnx": {"panns-onnx", "demucs-onnx", "faster-whisper", "noisereduce"}}.get(engine, {"noisereduce"})
    for m in status(s.model_dir):
        if m["key"] not in relevant:
            continue
        w = m["weights_installed"]
        if not m["python_deps_installed"]:
            state = "deps missing"
        elif w:
            state = "weights ready"
        elif not m["size_mb"]:
            state = "no weights needed"
        elif w is None:
            state = "downloads on first use"
        else:
            state = "weights not downloaded (audiolayer models download)"
        detail = f"{m['name']} — {state}"
        line(bool(m["python_deps_installed"]) and w is not False, str(m["key"]), detail)
    line(True, "workspace", str(s.workspace))
    ui = _ui_dist()
    line(ui is not None, "web UI", str(ui) if ui else "not built (cd frontend && npm ci && npm run build)")
    cfg = claude_desktop_config_path()
    has = False
    if cfg and cfg.exists():
        try:
            has = "audiolayer" in json.loads(cfg.read_text(encoding="utf-8")).get("mcpServers", {})
        except Exception:  # noqa: BLE001
            has = False
    line(has or None, "Claude Desktop", ("connected" if has else "not connected (audiolayer setup-claude)") + (f" · {cfg}" if cfg else ""))
    print("\nAll essentials OK." if ok else "\nSome essentials are missing — see above.")
    return 0 if ok else 1


def _ui_dist() -> Path | None:
    for p in (Path(__file__).resolve().parents[2] / "frontend" / "dist",
              Path(__file__).resolve().parent / "web_dist"):
        if (p / "index.html").exists():
            return p
    return None


# ---------------------------------------------------------------------------- models
def cmd_models(a: argparse.Namespace) -> int:
    from .models.download import default_keys, download
    from .models.registry import status

    s = get_settings()
    if a.action == "list":
        print(json.dumps(status(s.model_dir), indent=2))
        return 0
    download(s, a.keys or list(default_keys(s)))
    print("Done.")
    return 0


# ---------------------------------------------------------------------------- web
def cmd_web(a: argparse.Namespace) -> int:
    import uvicorn

    url = f"http://{a.host}:{a.port}"
    if not _ui_dist():
        _err("Web UI is not built; the API will run but the page will be empty. Run: cd frontend && npm ci && npm run build")
    if not a.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    print(f"AudioLayer Studio → {url}   (Ctrl+C to stop)\nWorkspace: {get_settings().workspace}")
    uvicorn.run("audiolayer.api.app:create_app", factory=True, host=a.host, port=a.port, log_level="warning")
    return 0


# ---------------------------------------------------------------------------- analyze / clean
def _analyze(a: argparse.Namespace) -> tuple[Any, Any]:
    from .core.schemas import AnalyzeOptions
    from .core.service import AudioLayerService

    src = Path(a.input).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"File not found: {src}")
    svc = AudioLayerService(get_settings())
    if not a.json:
        _err(f"Models: {svc.components.classifier.name} · "
             f"{getattr(svc.components.separator, 'name', 'no separator')} · "
             f"{getattr(svc.components.asr, 'name', 'no ASR')}")
    pr = svc.create_from_file(src, src.name, "local", move=False)
    opts = AnalyzeOptions(separate=not a.no_separate, transcribe=not a.no_transcribe, language=a.language)
    job = _wait(svc, svc.analyze(pr.id, opts, "local").id, "Analyzing", a.json)
    if job.status != "succeeded":
        svc.jobs.shutdown()
        raise SystemExit(f"Analysis {job.status}: {job.error}")
    return svc, svc.get(pr.id, "local")


def _summary(pr: Any) -> dict[str, Any]:
    a = pr.analysis
    return {
        "project_id": pr.id,
        "file": pr.media.filename,
        "duration_s": pr.media.audio_duration,
        "separated": a.separated,
        "loudness": a.loudness.model_dump(),
        "segments": [{"category": s.category, "start": round(s.start, 2), "end": round(s.end, 2),
                      "confidence": s.confidence, "uncertain": s.uncertain, "description": s.description,
                      "flags": s.flags, "transcript": s.transcript} for s in pr.segments],
        "issues": [{"kind": i.kind, "severity": i.severity, "title": i.title, "detail": i.detail} for i in a.issues],
        "limitations": a.limitations,
    }


def _print_summary(d: dict[str, Any]) -> None:
    print(f"\n{d['file']} · {d['duration_s']:.1f}s · loudness {d['loudness']['integrated_lufs']} LUFS"
          f" · separated: {d['separated']}")
    print("\nSegments:")
    for s in d["segments"]:
        unc = " (uncertain)" if s["uncertain"] else ""
        txt = f' “{s["transcript"][:60]}”' if s["transcript"] else ""
        flags = f" [{', '.join(s['flags'])}]" if s["flags"] else ""
        print(f"  {s['start']:7.2f}–{s['end']:7.2f}s  {s['category']:<8} {s['confidence']:.2f}{unc}  {s['description'][:60]}{txt}{flags}")
    if d["issues"]:
        print("\nProblems found:")
        for i in d["issues"]:
            print(f"  - [{i['severity']}] {i['title']}: {i['detail']}")
    if d["limitations"]:
        print("\nLimitations:")
        for x in d["limitations"]:
            print(f"  - {x}")


def cmd_analyze(a: argparse.Namespace) -> int:
    svc, pr = _analyze(a)
    try:
        d = _summary(pr)
        print(json.dumps(d, indent=2)) if a.json else _print_summary(d)
    finally:
        svc.jobs.shutdown()
    return 0


def cmd_clean(a: argparse.Namespace) -> int:
    from .core.schemas import ExportOptions

    remove = [c.strip() for c in (a.remove or "").split(",") if c.strip()]
    bad = [c for c in remove if c not in CATS]
    if bad:
        raise SystemExit(f"Unknown categories {bad}; choose from {', '.join(CATS)}")
    if a.project:
        from .core.service import AudioLayerService

        svc = AudioLayerService(get_settings())
        pr = svc.get(a.project, "local")  # reuse the analysis from `audiolayer analyze` (no re-processing)
        if pr.status != "analyzed":
            svc.jobs.shutdown()
            raise SystemExit(f"Project {a.project} is not analyzed (status: {pr.status})")
        pr = svc.apply_preset(pr.id, "clear", "local")  # start from a clean selection
    elif not a.input:
        raise SystemExit("Give an input file or --project <project_id>")
    else:
        svc, pr = _analyze(a)
    try:
        if a.mute_gibberish:
            pr = svc.apply_preset(pr.id, "remove_gibberish", "local")
        if a.preset:
            pr = svc.apply_preset(pr.id, a.preset, "local")  # keeps gibberish mutes
        if remove:
            pr = svc.set_removed(pr.id, sorted(set(remove) | set(pr.mix.removed)), "local")
        removed = list(pr.mix.removed)
        items: list[Any] = ["mix", "metadata"]
        if a.stems:
            items += list(CATS) + ["original"]
        if a.video and pr.media.has_video:
            items.append("video")
        opts = ExportOptions(items=items, format=a.format, remove=removed, loudness_preset=a.loudness, zip=a.zip)
        job = _wait(svc, svc.export(pr.id, opts, "local").id, "Exporting", a.json)
        if job.status != "succeeded":
            raise SystemExit(f"Export {job.status}: {job.error}")
        if a.out:
            out = Path(a.out).expanduser().resolve()
        elif a.input:
            src = Path(a.input).expanduser().resolve()
            out = src.parent / f"{src.stem}_audiolayer"
        else:
            out = Path.cwd() / f"{pr.name}_audiolayer"
        out.mkdir(parents=True, exist_ok=True)
        exp_dir = svc.store.path(pr.id, job.result["dir"])
        files = []
        for f in job.result["files"]:
            shutil.copy2(exp_dir / f, out / f)
            files.append(str(out / f))
        if job.result.get("zip"):
            z = svc.store.path(pr.id, job.result["zip"])
            shutil.copy2(z, out / z.name)
            files.append(str(out / z.name))
        d = _summary(pr)
        d.update({"removed": removed, "muted_segments": sum(s.muted for s in pr.segments), "output_dir": str(out),
                  "files": files, "export_limitations": job.result.get("limitations", []),
                  "skipped": job.result.get("skipped", []),
                  "mix_loudness_after": job.result.get("info", {}).get("mix_loudness_after")})
        if a.json:
            print(json.dumps(d, indent=2))
        else:
            _print_summary(d)
            print(f"\nRemoved: {', '.join(removed) or 'nothing'} · muted segments: {d['muted_segments']}")
            for x in d["export_limitations"]:
                print(f"  ! {x}")
            for s in d["skipped"]:
                print(f"  ! skipped {s['item']}: {s['reason']}")
            print(f"\nOutput → {out}")
            for f in files:
                print(f"  {Path(f).name}")
    finally:
        svc.jobs.shutdown()
    return 0


# ---------------------------------------------------------------------------- Claude setup
def claude_desktop_config_path() -> Path | None:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        return Path(appdata) / "Claude" / "claude_desktop_config.json" if appdata else None
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def mcp_server_entry(input_dirs: list[Path]) -> dict[str, Any]:
    s = get_settings()
    env = {
        "AUDIOLAYER_WORKSPACE": str(s.workspace),
        "AUDIOLAYER_MODEL_DIR": str(s.model_dir),
        "AUDIOLAYER_INPUT_DIRS": os.pathsep.join(str(p) for p in input_dirs),
    }
    path_parts = []
    ff = shutil.which("ffmpeg")
    if ff:
        path_parts.append(str(Path(ff).parent))
    if platform.system() == "Darwin":
        path_parts += ["/opt/homebrew/bin", "/usr/local/bin"]
    path_parts += [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    env["PATH"] = os.pathsep.join(dict.fromkeys(path_parts))
    return {"command": sys.executable, "args": ["-m", "audiolayer.mcp_server.server"], "env": env}


def cmd_setup_claude(a: argparse.Namespace) -> int:
    dirs = [Path(d).expanduser().resolve() for d in (a.input_dir or [])]
    if not dirs:
        default = Path.home() / "AudioLayer" / "inputs"
        dirs = [default]
        print(f"No --input-dir given; Claude will read media from {default}")
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    entry = mcp_server_entry(dirs)
    cfg = Path(a.config).expanduser() if a.config else claude_desktop_config_path()
    if a.desktop and cfg is not None:
        data: dict[str, Any] = {}
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text(encoding="utf-8") or "{}")
            except json.JSONDecodeError as e:
                raise SystemExit(f"{cfg} is not valid JSON ({e}); fix it or pass --config") from e
        data.setdefault("mcpServers", {})["audiolayer"] = entry
        if a.dry_run:
            print(json.dumps(data, indent=2))
        else:
            cfg.parent.mkdir(parents=True, exist_ok=True)
            if cfg.exists():
                bak = cfg.with_suffix(f".json.bak-{_dt.datetime.now():%Y%m%d-%H%M%S}")
                shutil.copy2(cfg, bak)
                print(f"Backup: {bak}")
            cfg.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"✓ Claude Desktop configured: {cfg}\n  Quit Claude Desktop completely and open it again.")
    code_cmd = ["claude", "mcp", "add", "audiolayer", "--scope", "user"]
    for k in ("AUDIOLAYER_WORKSPACE", "AUDIOLAYER_MODEL_DIR", "AUDIOLAYER_INPUT_DIRS"):
        code_cmd += ["-e", f"{k}={entry['env'][k]}"]
    code_cmd += ["--", entry["command"], *entry["args"]]
    if a.code:
        if shutil.which("claude") and not a.dry_run:
            r = subprocess.run(code_cmd, check=False)
            print("✓ Claude Code configured" if r.returncode == 0 else "Claude Code setup failed (see above)")
        else:
            print("Claude Code command:\n  " + subprocess.list2cmdline(code_cmd))
    print(f"\nMedia folder(s) Claude may read: {', '.join(map(str, dirs))}")
    print('Try in Claude: "Analyze <file> in my AudioLayer inputs folder and remove the background music."')
    return 0


def cmd_mcp(_: argparse.Namespace) -> int:
    from .mcp_server.server import main as mcp_main

    mcp_main(["--transport", "stdio"])
    return 0


# ---------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audiolayer", description="AudioLayer Studio — clean up AI-video audio.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check installation").set_defaults(fn=cmd_doctor)

    m = sub.add_parser("models", help="list or download model weights")
    m.add_argument("action", choices=["list", "download"])
    m.add_argument("keys", nargs="*")
    m.set_defaults(fn=cmd_models)

    w = sub.add_parser("web", help="start the web app and open the browser")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--no-browser", action="store_true")
    w.set_defaults(fn=cmd_web)

    def common(sp: argparse.ArgumentParser, optional_input: bool = False) -> None:
        sp.add_argument("input", nargs="?" if optional_input else None, help="video or audio file")
        sp.add_argument("--no-separate", action="store_true", help="detection only (faster)")
        sp.add_argument("--no-transcribe", action="store_true", help="skip speech recognition")
        sp.add_argument("--language", help="speech language hint, e.g. en, id")
        sp.add_argument("--json", action="store_true", help="machine-readable output")

    an = sub.add_parser("analyze", help="report what is in the audio")
    common(an)
    an.set_defaults(fn=cmd_analyze)

    cl = sub.add_parser("clean", help="analyze, remove categories and export in one go")
    common(cl, optional_input=True)
    cl.add_argument("--project", help="reuse a project_id printed by `audiolayer analyze` (skips re-analysis)")
    cl.add_argument("--remove", help=f"comma list of categories to remove: {', '.join(CATS)}")
    cl.add_argument("--preset", help=f"cleanup preset: {PRESET_HELP}")
    cl.add_argument("--mute-gibberish", action="store_true", help="mute dialogue flagged as unintelligible")
    cl.add_argument("--loudness", default="none", choices=["none", "youtube", "tiktok", "podcast", "broadcast"])
    cl.add_argument("--format", default="wav", choices=["wav", "flac", "mp3"])
    cl.add_argument("--video", action="store_true", help="also write the video with the cleaned audio")
    cl.add_argument("--stems", action="store_true", help="also export all five stems + original audio")
    cl.add_argument("--zip", action="store_true", help="also write a ZIP package")
    cl.add_argument("--out", help="output folder (default: <input>_audiolayer next to the input)")
    cl.set_defaults(fn=cmd_clean)

    sc = sub.add_parser("setup-claude", help="connect AudioLayer to Claude Desktop / Claude Code")
    sc.add_argument("--input-dir", action="append", help="folder Claude may read media from (repeatable)")
    sc.add_argument("--config", help="path to claude_desktop_config.json (auto-detected)")
    sc.add_argument("--no-desktop", dest="desktop", action="store_false", help="skip Claude Desktop")
    sc.add_argument("--code", action="store_true", help="also register with Claude Code (`claude mcp add`)")
    sc.add_argument("--dry-run", action="store_true", help="print the config instead of writing it")
    sc.set_defaults(fn=cmd_setup_claude)

    sub.add_parser("mcp", help="run the MCP server on stdio").set_defaults(fn=cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
