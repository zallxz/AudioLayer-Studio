---
name: audiolayer
description: Clean up the audio of AI-generated videos (Veo, Sora, Kling, Wan, Runway…) with AudioLayer Studio — detect dialogue, background music, sound effects, songs and noise, then remove or isolate them, mute gibberish voices, make a music-and-effects track, fix loudness and export synced stems or the video with cleaned audio. Use whenever the user wants to remove/keep/separate music, voices, SFX or noise in a video or audio file.
---

# AudioLayer — AI-video audio cleanup

AudioLayer Studio (https://github.com/zallxz/AudioLayer-Studio): FFmpeg + Demucs (separation) + PANNs
(sound-event detection) + Whisper (speech). It never modifies the original file. It runs either on the
user's computer (MCP / CLI) or **directly in Claude's own code sandbox with nothing installed by the
user** (torch-free "cloud" engine, section 5).
Categories: `dialogue`, `bgm` (music under the scene), `sfx` (effects + intentional ambience),
`music` (foreground song/performance), `noise` (hiss, hum, rumble — never ambience).

## 1. Pick the route

1. **MCP tools available** (`analyze_media`, `get_processing_status`, … from the `audiolayer` server)
   → use them (section 4).
2. **The user attached/uploaded a media file and you can run code** (Claude app/web with code
   execution, a cloud sandbox) → **cloud mode** (section 5): install AudioLayer into the sandbox
   yourself, process the uploaded file there, hand back the outputs. The user installs nothing.
3. **A shell on the user's own machine** (Claude Code) → use the CLI (section 6). Check first:
   `audiolayer --version`.
4. **None of these** (no tools, no code execution) → explain the options (section 8); don't pretend to
   process audio.

## 2. Default flow: detect → ask → apply

When the user gives a file without saying exactly what to remove, don't ask anything first:

1. **Detect right away** — run the analysis (MCP `analyze_media`, or `audiolayer analyze "<file>" --json`).
   Tell the user in one line that it's running and roughly how long (≈1.5× the clip length on CPU).
2. **Show what was found**, briefly, in the user's language: each category present with its time ranges
   (e.g. "Musik latar 0:00–0:20", "Dialog 0:03–0:11: “With her white paint…”"), dialogue flagged as
   gibberish, uncertain segments, and audio problems (hiss, hum, clipping, loudness).
3. **Ask with multiple choice**, built only from what was actually detected (use the ask-user /
   multiple-choice tool if the app has one; otherwise a short numbered list the user can answer with
   numbers):
   - *What should be removed?* (multi-select) — e.g. "Background music (0:00–0:20)",
     "Gibberish dialogue (2 segments)", "All dialogue (keep music & effects)", "Sound effects: birds
     0:13–0:15", "Noise/hiss", "Nothing — just give me the stems".
   - *What do you want back?* — cleaned video · cleaned audio only · stems for my editor.
   - *Platform loudness?* — YouTube/TikTok (−14 LUFS) · keep as is.
   Mark the option you recommend (e.g. remove music + noise when the user complained about music).
4. **Apply and export** with the answers (MCP `remove_audio_category` + `export_cleaned_audio` /
   `export_audio_stems`, or one `audiolayer clean … --remove … --video --loudness …` command), then report
   what changed, the output files, and the limitations.

If the request already says what to do ("hapus musiknya, export buat TikTok"), skip step 3 and act,
then still report what was found and removed. If the user corrects a label ("that's a song, not BGM"),
relabel it before exporting.

## 3. Translate the request

| User says (EN / ID) | Do |
|---|---|
| "remove the music", "no BGM", "hapus musik/lagu latar" | preset `remove_music` (removes `bgm` + `music`) |
| "keep only the voice", "dialog saja", "suara orang saja" | preset `dialogue_only` |
| "remove the voice / I'll dub it", "hapus suara orang", "buat dubbing" | preset `music_and_effects` (M&E) |
| "the characters babble / gibberish / fake language", "ngoceh", "bahasa aneh" | `--mute-gibberish` / preset `remove_gibberish` (mutes only flagged dialogue) |
| "remove hiss/hum/noise", "hapus desis/dengung" | preset `remove_noise` |
| "remove the sound effects", "hapus efek suara" | preset `remove_sfx` |
| "for YouTube/TikTok/Reels", "volume sama antar klip" | loudness `youtube`/`tiktok` (−14 LUFS) |
| "give me the parts / stems for my editor" | export stems (+ metadata, ZIP) |
| "put it back in the video" | include video in the export |
| unclear ("clean this up", "bersihkan") | analyze first, report what's there, propose a preset, then act |

Several wishes combine: e.g. remove music **and** mute gibberish **and** normalize for YouTube.

## 4. MCP workflow

1. `analyze_media(file_path=...)` → returns `project_id`, `job_id`. If you get `[forbidden] … outside
   the allowed input folders`, ask the user to put the file in the folder named in the hint (or run
   `audiolayer setup-claude --input-dir <folder>` and restart Claude Desktop).
2. Poll `get_processing_status(job_id)` every few seconds until `succeeded`/`failed` (roughly 1.5× the
   clip length on a CPU). Don't claim results before it succeeds.
3. `get_analysis` and `list_audio_segments` → summarize for the user: what is where (times), problems
   found, segments flagged `uncertain`, `role_uncertain`, `unintelligible`.
4. Fix labels the user corrects with `update_segment_category` (e.g. a "BGM" that is really an on-screen song → `music`).
5. `remove_audio_category(project_id, preset=... | categories=[...])` → relay the returned `limitations`.
6. `export_cleaned_audio(project_id, loudness_preset=..., include_video=true/false)` and/or
   `export_audio_stems(project_id)` / `separate_audio_category(project_id, category)`.
7. Poll the export job; give the user the file paths from `files` / `zip`.

## 5. Cloud mode — run in Claude's sandbox (nothing for the user to install)

Setup (once per conversation; ~1–2 min; downloads ≈300 MB of PyPI wheels and ≈400 MB of model weights
from GitHub — no PyTorch, no Hugging Face):

```bash
audiolayer --version 2>/dev/null || pip install -q "audiolayer-studio[cloud] @ https://github.com/zallxz/AudioLayer-Studio/archive/refs/heads/main.zip"
# if pip refuses ("externally managed"): add --break-system-packages
audiolayer models download      # ONNX models + Whisper from the GitHub mirror, SHA-256 verified
audiolayer doctor               # expect "AI engine: ONNX Runtime (torch-free cloud engine)"
```

Then run the normal flow (section 2) on the uploaded file — find it where the app stores uploads (e.g.
`/mnt/user-data/uploads/`), and write results where the user can download them (e.g. `--out
/mnt/user-data/outputs/<name>`), then share/present those files:

```bash
audiolayer analyze "/mnt/user-data/uploads/clip.mp4" --json     # detect → summarize → ask (note project_id)
audiolayer clean --project <project_id> --remove bgm,music --mute-gibberish \
  --loudness youtube --video --out /mnt/user-data/outputs/clip_clean   # apply choices, no re-analysis
```

Notes: processing is ~1.5–2× the clip length on the sandbox CPU — use generous command timeouts and
`--no-transcribe` for long clips if speed matters. FFmpeg comes from the `imageio-ffmpeg` wheel when the
sandbox has none. If `pip install` or the model download fails with a network error, code execution or
its network access is turned off: tell the user to enable *Code execution* and allow at least package
managers (PyPI + GitHub) in *Settings → Capabilities*, then retry. Uploaded files are processed only
inside the sandbox; nothing is sent to other services.

## 6. CLI workflow

```bash
audiolayer analyze "clip.mp4"                       # what's in the audio (add --json to parse)
audiolayer clean --project <project_id> --preset remove_music --video   # reuse that analysis
audiolayer clean "clip.mp4" --preset remove_music --mute-gibberish --loudness youtube --video
audiolayer clean "clip.mp4" --remove sfx,noise --stems --zip --out ./cleaned
audiolayer doctor                                   # diagnose installation problems
```

Options: `--remove` (comma list of categories), `--preset`, `--mute-gibberish`,
`--loudness none|youtube|tiktok|podcast|broadcast`, `--format wav|flac|mp3`, `--video`, `--stems`,
`--zip`, `--out DIR`, `--project ID` (reuse an analysis), `--no-transcribe` (faster), `--language id|en|…`,
`--json`.
Output goes to `<input>_audiolayer/` next to the input by default. Use `--json` when you need to read
results programmatically; run long jobs with a generous timeout (minutes for long clips).

**Not installed?** Ask before installing on the user's own computer (≈2.5 GB with PyTorch + models).
Easiest: `git clone https://github.com/zallxz/AudioLayer-Studio && cd AudioLayer-Studio && ./install.sh`
(Windows: `powershell -ExecutionPolicy Bypass -File install.ps1`). In a throwaway sandbox:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # if blocked: pip install torch
pip install "audiolayer-studio[ml] @ git+https://github.com/zallxz/AudioLayer-Studio"
audiolayer doctor
```

FFmpeg must be on PATH. Model weights download on first use from zenodo.org, huggingface.co and
dl.fbaipublicfiles.com — if the network blocks them, AudioLayer falls back to detection-only with a
low-accuracy heuristic classifier; say so plainly instead of presenting those results as reliable.

## 7. Report honestly

- Lead with what was found (categories + time ranges, transcript snippets, problems) and what was removed.
- Always pass on the `limitations` list. Known weak spots: music and SFX that sound **at the same time**
  are only approximately split (`overlaps_music_split_estimated`); percussive SFX can be detected as
  music; BGM vs. foreground music is a judgment (`role_uncertain`); gibberish detection is a confidence
  hint; clipping and muffled (band-limited) audio cannot be repaired.
- Suggest the user listens (web app A/B toggle: `audiolayer web`) before publishing.
- Never say the separation is perfect or lossless; never invent outputs or timings.

## 8. Setup help (when nothing is installed)

1. Install: clone the repo and run `./install.sh` (macOS/Linux) or `install.ps1` (Windows). It installs
   everything, downloads models, creates a launcher and connects Claude Desktop (and Claude Code if present).
2. Put clips in `~/AudioLayer/inputs` (the folder Claude may read), restart Claude Desktop.
3. Web app: `audiolayer web` or double-click `AudioLayer.command` / `AudioLayer.bat`.
4. In the Claude app/web, the easiest path is cloud mode (section 5): just upload the clip — no install.
   (Claude on the web cannot reach a program on the user's laptop; for local files use Claude Desktop.)
Full tutorial in the repo: docs/TUTORIAL.en.md (English) and docs/TUTORIAL.md (Bahasa Indonesia).
