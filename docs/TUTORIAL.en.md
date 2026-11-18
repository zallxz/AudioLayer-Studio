# AudioLayer Studio Tutorial

[🇮🇩 Bahasa Indonesia](TUTORIAL.md) · **🇬🇧 English**

A step-by-step guide from zero to cleaning up the audio of AI-generated videos — in the web app on
your own computer **and** through Claude.

---

## Zero-install: use it inside Claude

🎬 Watch first: [audiolayer_promo.mp4](media/audiolayer_promo.mp4) (41 s, includes the Claude walkthrough, Indonesian voice-over).

1. In the Claude app (web, desktop or mobile) go to **Settings → Capabilities** and turn on
   **Code execution**. Network access must allow **PyPI** (package managers) **and GitHub**
   (`github.com`, `raw.githubusercontent.com`) — the AI models download from this repo. If you only
   allowed *package managers* and the model download fails, add those two GitHub domains or pick *All domains*.
2. **Download the skill: [audiolayer.zip](https://github.com/zallxz/AudioLayer-Studio/raw/main/skills/audiolayer.zip)**, then in **Settings → Capabilities → Skills** click
   **Upload skill** and choose that file.
3. Start a new chat, **upload your clip** and type **`/audiolayer`** (or just ask, e.g.
   *"remove the background music from this video"*).

Claude installs AudioLayer's torch-free engine in its own sandbox (takes 1–2 minutes the first time),
tells you what it found (music, dialogue, effects, noise, gibberish…), **asks what to remove**, then
gives you the cleaned video/audio files to download. Nothing is installed on your computer.
Best for short clips (the app's upload limit applies).

What it looks like:

> **You:** *(upload `clip.mp4`)* `/audiolayer`
> **Claude:** Here's what's in your clip: background music 0:00–0:20 · dialogue 0:03–0:12 “With her white paint…” ·
> bird sound effect 0:13–0:15 · hiss 0:00–0:20. **What should be removed?** ☐ Background music ☐ Hiss ☐ Sound effects ☐ Nothing — just stems
> **You:** *(tick Background music + Hiss)*
> **Claude:** Done — `clip_cleaned_video.mp4`, `clip_cleaned.wav` ready to download.

**If it fails:** a *network error* during `pip install` or the model download means Code execution is off
or its network access doesn't allow PyPI/GitHub yet (see step 1). Processing takes ≈1.5–2× the clip
length; for long clips ask Claude to use `--no-transcribe` to speed it up.

---

## ⚡ Fastest way on your own computer (3 steps)

1. **Install the basics**: Python 3.11+, Git and FFmpeg ([step 2](#2-install-the-basics)).
   The installer below can install Python/FFmpeg for you via Homebrew (Mac), apt (Ubuntu) or winget (Windows).
2. **Download and run the installer**:
   ```bash
   git clone https://github.com/zallxz/AudioLayer-Studio.git
   cd AudioLayer-Studio
   ./install.sh                                          # macOS / Linux
   ```
   ```powershell
   powershell -ExecutionPolicy Bypass -File install.ps1  # Windows
   ```
   The installer creates a virtual environment, installs PyTorch + AudioLayer, downloads the AI models
   (~550 MB), creates a launcher, and **connects Claude Desktop** (+ Claude Code if installed).
   Node.js is not required — the web UI ships prebuilt.
3. **Use it**:
   - **Web app** → double-click `AudioLayer.command` (Mac/Linux) or `AudioLayer.bat` (Windows).
   - **One command** → `audiolayer clean video.mp4 --preset remove_music --loudness youtube --video`
   - **Claude** → put videos in the `AudioLayer/inputs` folder (in your home folder), restart Claude
     Desktop, then type e.g. *"remove the background music from scene1.mp4"*.

Check anytime that everything is fine: `audiolayer doctor`.

The steps below are the manual, detailed version of the same process.

---

## Contents

1. [Requirements](#1-requirements)
2. [Install the basics (Python, FFmpeg, Node.js, Git)](#2-install-the-basics)
3. [Download & install AudioLayer](#3-download--install-audiolayer)
4. [Download the AI models](#4-download-the-ai-models-once)
5. [Use the web app](#5-use-the-web-app)
6. [Use with Claude Desktop](#6-use-with-claude-desktop)
7. [Use with Claude Code](#7-use-with-claude-code-terminal)
8. [Does it work on claude.ai (web)?](#8-does-it-work-on-claudeai-web) — yes, see Zero-install
9. [Quick recipes for common AI-video problems](#9-quick-recipes)
   · [`audiolayer` commands (CLI)](#audiolayer-commands-cli) · [Claude skill](#claude-skill)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11, macOS 12+, Linux | — |
| RAM | 8 GB | 16 GB |
| Free disk | ~4 GB (app + AI models) | more for many videos |
| GPU | not required | NVIDIA (CUDA) → much faster |
| Internet | for installing & first model download | works offline afterwards |

Speed on a typical laptop CPU: about **1.5× the video length** (a 20 s clip ≈ 30 s of processing).
Everything runs on your own computer — your videos are not uploaded anywhere.

---

## 2. Install the basics

### Windows

1. **Python 3.11 or newer** — download from <https://www.python.org/downloads/>.
   During setup, **tick "Add python.exe to PATH"**.
2. **FFmpeg** — open *PowerShell* and run:
   ```powershell
   winget install Gyan.FFmpeg
   ```
3. **Node.js 20+** (only needed to change the UI): `winget install OpenJS.NodeJS.LTS`
4. **Git**: `winget install Git.Git`
5. **Close and reopen** PowerShell, then check:
   ```powershell
   python --version
   ffmpeg -version
   git --version
   ```

### macOS

1. Install Homebrew (if you don't have it): see <https://brew.sh>
2. Then in Terminal:
   ```bash
   brew install python@3.12 ffmpeg git
   ```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg libsndfile1 git
```

---

## 3. Download & install AudioLayer

Open Terminal (macOS/Linux) or PowerShell (Windows):

```bash
git clone https://github.com/zallxz/AudioLayer-Studio.git
cd AudioLayer-Studio
```

### 3a. Create a Python virtual environment

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
> If Windows refuses to run the script: run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

When it works, your prompt starts with `(.venv)`. **Every time** you open a new terminal to use
AudioLayer, activate it again with the `activate` command above.

### 3b. Install PyTorch (the AI engine)

Pick **one**:

```bash
# PC/laptop without an NVIDIA GPU (Windows / Linux):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# macOS (Apple Silicon / Intel):
pip install torch

# PC with an NVIDIA GPU: get the exact command from https://pytorch.org/get-started/locally/
```

### 3c. Install AudioLayer

```bash
pip install -e ".[ml]"
```

### 3d. Build the web UI (optional)

A ready-to-use web UI is already included. You only need this if you change the UI code:

```bash
cd frontend
npm ci
npm run build
cd ..
```

---

## 4. Download the AI models (once)

```bash
audiolayer models download
```

This downloads ~550 MB: PANNs (sound detection), Demucs (source separation), Whisper (transcription).
If you skip this, the models download automatically on the first analysis.

Check model status:
```bash
audiolayer models list
```

**Want to try without your own video?** Create a 20-second demo clip (speech + music + effects + noise):
```bash
python scripts/make_demo_media.py demo
```
Result: `demo/demo_ai_video.mp4`.

---

## 5. Use the web app

### Start it

```bash
audiolayer web
```

Your browser opens **<http://127.0.0.1:8000>** automatically (or use the double-click launcher).
Keep the terminal open; press `Ctrl + C` to stop.

### Workflow

1. **Upload** — click *Upload media* or drag a file (MP4, MOV, MKV, WebM, WAV, MP3, AAC, FLAC) onto the page.
   Analysis starts right away; progress shows in the top bar (you can *Cancel*).
2. **Read the results** — the timeline shows the waveform and 5 colored lanes:
   **Dialogue** (blue), **BGM** (purple), **SFX** (orange), **Music** (pink), **Noise** (grey).
   - More opaque box = more confident. Hatched box = *uncertain* (check it yourself).
   - A red **?!** on dialogue = gibberish / unintelligible speech.
   - Left panel: **Detected problems** (hiss, hum, clipping, loudness, …) with fix buttons.
3. **Listen** — press ▶ or `Space`. The category buttons in the bottom bar (Dialogue, BGM, …) turn a
   category off/on; what you hear is exactly what will be exported. **A/B original** compares with the original.
4. **Fix labels** — click a box on the timeline → the right panel shows its details. If the category is
   wrong (e.g. a song detected as BGM), click the right one. You can also:
   - **Isolate**: hear only that segment.
   - **Remove segment**: remove just that part.
   - **Remove all …**: remove the whole category.
   - Edit *Start/End* to tidy the time range.
   - Made a mistake? **↶ / ↷** (undo/redo) at the top.
5. **Quick fixes** (left panel) — one click for common cases: *Remove all music*, *Dialogue only*,
   *Music & Effects (M&E)*, *Remove noise*, *Remove sound effects*, *Mute gibberish dialogue*.
6. **Export** — yellow *Export* button:
   - Pick files: cleaned mix, a stem per category, original audio, **video with cleaned audio**, metadata.
   - Format: WAV 48 kHz 24-bit / FLAC / MP3.
   - **Loudness**: YouTube/TikTok −14 LUFS, Podcast −16, Broadcast −23 (makes stitched clips match).
   - Click *Export* → *Download ZIP*.

All stems have the same length and start at 0 s — drop them at the start of a Premiere/DaVinci/CapCut/
Audacity timeline and they line up. `*_audacity_labels.txt` can be imported in Audacity
(*File → Import → Labels*) to see segment markers.

Output files are also kept in `AudioLayer/workspace/projects/<id>/exports/` in your home folder.

---

## 6. Use with Claude Desktop

**Yes, you can use AudioLayer directly from Claude.** Claude Desktop runs AudioLayer on your computer
via MCP, so you just ask Claude in plain language.

### Automatic (recommended)

```bash
audiolayer setup-claude --input-dir "~/Documents/AI Videos"         # add --code for Claude Code too
```
This writes the Claude Desktop config for you (your old file is backed up first). Restart Claude
Desktop and jump to [Step 5](#step-5--ask-claude). `install.sh`/`install.ps1` already ran it with the
`~/AudioLayer/inputs` folder.

### Manual

#### Step 1 — Find the `audiolayer-mcp` program

With the venv active:

macOS / Linux:
```bash
which audiolayer-mcp
# e.g. /Users/name/AudioLayer-Studio/.venv/bin/audiolayer-mcp
```
Windows:
```powershell
where.exe audiolayer-mcp
# e.g. C:\Users\name\AudioLayer-Studio\.venv\Scripts\audiolayer-mcp.exe
```

#### Step 2 — Choose your video folder

Claude may only read videos from folders you allow (for safety). For example, create an
`AI Videos` folder in Documents and keep your clips there.

#### Step 3 — Edit the Claude Desktop config

Open Claude Desktop → **Settings → Developer → Edit Config**. The file is:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Content (adjust the paths):

**macOS / Linux**
```json
{
  "mcpServers": {
    "audiolayer": {
      "command": "/Users/name/AudioLayer-Studio/.venv/bin/audiolayer-mcp",
      "env": {
        "AUDIOLAYER_WORKSPACE": "/Users/name/AudioLayer/workspace",
        "AUDIOLAYER_INPUT_DIRS": "/Users/name/Documents/AI Videos",
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
```
> `PATH` lets Claude find FFmpeg installed by Homebrew.

**Windows** (backslashes must be doubled `\\`)
```json
{
  "mcpServers": {
    "audiolayer": {
      "command": "C:\\Users\\name\\AudioLayer-Studio\\.venv\\Scripts\\audiolayer-mcp.exe",
      "env": {
        "AUDIOLAYER_WORKSPACE": "C:\\Users\\name\\AudioLayer\\workspace",
        "AUDIOLAYER_INPUT_DIRS": "C:\\Users\\name\\Documents\\AI Videos"
      }
    }
  }
}
```

If the config already has other MCP servers, add the `"audiolayer": {...}` block inside the existing
`"mcpServers"` (separated by a comma). Several video folders: separate them with `:` (macOS/Linux)
or `;` (Windows).

#### Step 4 — Restart Claude Desktop

Quit Claude Desktop completely (including from the tray/menu bar) and open it again. The tools/connectors
menu in the chat box now lists **audiolayer**.

### Step 5 — Ask Claude

Example requests:

- *"Analyze `scene1.mp4` in my AI Videos folder with AudioLayer. What sounds are in it?"*
- *"Remove the background music but keep the dialogue and sound effects, then export the video with
  clean audio for YouTube."*
- *"Is there any gibberish dialogue? If so, mute only those parts."*
- *"Make a Music & Effects version (no voices) so I can re-dub it."*
- *"Check the audio quality: any hum, hiss, clipping or too-quiet volume?"*
- *"The music segment at 12 s is actually a song, not BGM — fix the label."*
- *"Export all stems as WAV."*

Claude runs the analysis, waits until it's done, reports the results (including limitations and
uncertain parts), and gives you the file locations on your computer.

**Bonus:** projects created through Claude also appear in the web app (same `AUDIOLAYER_WORKSPACE`) — let
Claude analyze, then check and listen on the timeline.

### Tools available to Claude

| Tool | Purpose |
|---|---|
| `analyze_media` | Start analyzing a video/audio file |
| `get_processing_status` / `cancel_processing` | Check progress / cancel |
| `get_analysis`, `get_quality_report` | Results summary & audio problems |
| `list_audio_segments` | Segments (time, category, confidence, transcript) |
| `update_segment_category` | Change label / times / mute a segment |
| `remove_audio_category` | Choose categories to remove or use a preset |
| `separate_audio_category` | Extract one category as a file |
| `export_audio_stems`, `export_cleaned_audio` | Export stems / cleaned audio (+ video) |
| `list_projects`, `get_system_info` | Project & model info |

---

## 7. Use with Claude Code (terminal)

Easiest (venv active):

```bash
audiolayer setup-claude --input-dir "$HOME/Documents/AI Videos" --code
```

Or manually:

```bash
claude mcp add audiolayer --scope user \
  -e AUDIOLAYER_INPUT_DIRS="$HOME/Documents/AI Videos" \
  -e AUDIOLAYER_WORKSPACE="$HOME/AudioLayer/workspace" \
  -- "$(which audiolayer-mcp)"
```

Then in Claude Code: *"use audiolayer to remove the music from ~/Documents/AI Videos/scene1.mp4"*.
Claude Code can also call the CLI directly (`audiolayer clean …`).

---

## 8. Does it work on claude.ai (web)?

**Yes — with the skill (cloud mode).** Upload the clip in the chat and Claude processes it in its own
sandbox (see [Zero-install](#zero-install-use-it-inside-claude) at the top). What claude.ai cannot do
is reach a program running on your laptop (`localhost`), so for files on your computer:

- **Easiest:** use **Claude Desktop** (step 6) — free download from <https://claude.ai/download>.
- **Advanced:** run AudioLayer on your own server with a public HTTPS address and add it as a
  *custom connector* on claude.ai (if your plan/organization allows it). Guide:
  [docs/MCP.md → Remote deployment](MCP.md#remote-deployment).

---

## 9. Quick recipes

| AI-video problem | What to do |
|---|---|
| Music you didn't ask for | Quick fix **Remove all music** → Export |
| Characters speak gibberish / a fake language | **Mute gibberish dialogue**, or **Music & Effects** to drop all dialogue |
| Replace the AI voice with your own voice-over | **Music & Effects (M&E)** → export, add the new voice in your editor |
| Need dialogue only (subtitles / dubbing) | **Dialogue only**, export the Dialogue stem + `.srt` |
| Hiss / hum | **Remove noise** |
| Stitched clips have different volumes | Export every clip with loudness **YouTube −14 LUFS** + 5 ms edge fade |
| Audio sounds muffled | Reported as *Band-limited audio* — separation can't restore it; regenerate or use a bandwidth-extension tool |
| Video has no sound | Reported as *No audio track / Silent audio* — regenerate with audio enabled |

**Be aware (honestly):** separating voices from everything else works well; but music and sound
effects that play **at the same time** can't yet be split cleanly (flagged "split is estimated").
Drum-like effects are sometimes detected as music. Always listen with **A/B original** before
exporting. Details: [docs/MODELS.md](MODELS.md).

### `audiolayer` commands (CLI)

| Command | Purpose |
|---|---|
| `audiolayer web` | Open the web app in your browser |
| `audiolayer analyze video.mp4` | Show what's in the audio (segments, transcript, problems) |
| `audiolayer clean video.mp4 --preset remove_music --video` | Remove music, save the cleaned video |
| `audiolayer clean video.mp4 --remove sfx,noise --stems --zip` | Remove SFX & noise, save all stems + ZIP |
| `audiolayer clean video.mp4 --mute-gibberish --loudness tiktok` | Mute gibberish dialogue, TikTok loudness |
| `audiolayer setup-claude --input-dir <folder>` | Connect to Claude Desktop (`--code` for Claude Code) |
| `audiolayer models download` | Download the AI models |
| `audiolayer doctor` | Check the installation |

`clean` writes its results to a `<video-name>_audiolayer` folder next to the original video.
Presets: `remove_music`, `dialogue_only`, `music_and_effects`, `remove_noise`, `remove_sfx`, `remove_gibberish`.

### Claude skill

The repo includes a Claude skill at [`skills/audiolayer/SKILL.md`](../skills/audiolayer/SKILL.md).
It teaches Claude when and how to use AudioLayer: turning requests ("remove the music", "they speak
gibberish", "I want to dub it") into the right preset, running the analysis, waiting for results, and
reporting limitations honestly.

- **Claude (app/web)**: *Settings → Capabilities → Skills → Upload skill*, then choose a ZIP containing
  the `audiolayer/` folder (or save it from a skill card Claude offers you).
- **Claude Code**: copy the folder to `~/.claude/skills/audiolayer/` (all projects) or
  `.claude/skills/audiolayer/` (one project).

The skill is most useful together with the MCP server (Claude Desktop) or the CLI (Claude Code).
Without either, it guides the user through installing first.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| Not sure what's wrong | Run `audiolayer doctor` — it checks every part one by one |
| `audiolayer` / `audiolayer-api` / `audiolayer-mcp` *not found* | The venv isn't active — run the `activate` command again (step 3a), or use the `AudioLayer.command`/`.bat` launcher |
| `FFmpeg not found on PATH` | Install FFmpeg (step 2) and reopen the terminal. For Claude Desktop on macOS add `PATH` to the config (step 6) |
| Claude Desktop doesn't list audiolayer | Check the `command` path is correct & absolute; valid JSON (commas, `\\` on Windows); restart fully. Logs: macOS `~/Library/Logs/Claude/mcp-server-audiolayer.log`, Windows `%APPDATA%\Claude\logs` |
| Claude says `Path is outside the allowed input folders` | Move the video into a folder in `AUDIOLAYER_INPUT_DIRS`, or add the folder |
| "No source-separation model available" | PyTorch / `.[ml]` not installed (steps 3b–3c) |
| Model download fails | Check your internet and run `audiolayer models download` again |
| Blank web page / 404 | Run `audiolayer doctor`; if the *web UI* line fails, reinstall (`pip install -e ".[ml]"`) |
| No sound in the browser | Click ▶ once (browsers block audio until you click); make sure the status is *analyzed* |
| Claude can't install AudioLayer in its sandbox | Turn on *Code execution* and allow package managers (PyPI + GitHub) in *Settings → Capabilities* |
| Slow / hot laptop | Normal on CPU. Turn off *Transcribe dialogue*, or set `AUDIOLAYER_WHISPER_MODEL=tiny` |
| Out of memory on long videos | Set `AUDIOLAYER_CHUNK_SECONDS=15` |

Still stuck? Open an *Issue* at <https://github.com/zallxz/AudioLayer-Studio/issues> with the error
message and your OS.
