---
name: audiolayer
description: Clean up the audio of AI-generated videos (Veo, Sora, Kling, Wan, Runway…) with AudioLayer Studio — detect dialogue, background music, sound effects, songs and noise, then remove or isolate them, mute gibberish voices, make a music-and-effects track, fix loudness and export synced stems or the video with cleaned audio. Use whenever the user wants to remove/keep/separate music, voices, SFX or noise in a video or audio file.
---

# AudioLayer — AI-video audio cleanup

AudioLayer Studio (https://github.com/zallxz/AudioLayer-Studio) runs locally: FFmpeg + Demucs
(separation) + PANNs (sound-event detection) + Whisper (speech). It never modifies the original file.
Categories: `dialogue`, `bgm` (music under the scene), `sfx` (effects + intentional ambience),
`music` (foreground song/performance), `noise` (hiss, hum, rumble — never ambience).

## 1. Pick the route

1. **MCP tools available** (`analyze_media`, `get_processing_status`, … from the `audiolayer` server)
   → use them (section 3).
2. **A shell is available** (Claude Code, or a sandbox with the user's uploaded file) → use the CLI
   (section 4). Check first: `audiolayer --version`.
3. **Neither** → tell the user how to set it up (section 6) and stop; don't pretend to process audio.

## 2. Translate the request

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

## 3. MCP workflow

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

## 4. CLI workflow

```bash
audiolayer analyze "clip.mp4"                       # what's in the audio (add --json to parse)
audiolayer clean "clip.mp4" --preset remove_music --mute-gibberish --loudness youtube --video
audiolayer clean "clip.mp4" --remove sfx,noise --stems --zip --out ./cleaned
audiolayer doctor                                   # diagnose installation problems
```

Options: `--remove` (comma list of categories), `--preset`, `--mute-gibberish`,
`--loudness none|youtube|tiktok|podcast|broadcast`, `--format wav|flac|mp3`, `--video`, `--stems`,
`--zip`, `--out DIR`, `--no-transcribe` (faster), `--language id|en|…`, `--json`.
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

## 5. Report honestly

- Lead with what was found (categories + time ranges, transcript snippets, problems) and what was removed.
- Always pass on the `limitations` list. Known weak spots: music and SFX that sound **at the same time**
  are only approximately split (`overlaps_music_split_estimated`); percussive SFX can be detected as
  music; BGM vs. foreground music is a judgment (`role_uncertain`); gibberish detection is a confidence
  hint; clipping and muffled (band-limited) audio cannot be repaired.
- Suggest the user listens (web app A/B toggle: `audiolayer web`) before publishing.
- Never say the separation is perfect or lossless; never invent outputs or timings.

## 6. Setup help (when nothing is installed)

1. Install: clone the repo and run `./install.sh` (macOS/Linux) or `install.ps1` (Windows). It installs
   everything, downloads models, creates a launcher and connects Claude Desktop (and Claude Code if present).
2. Put clips in `~/AudioLayer/inputs` (the folder Claude may read), restart Claude Desktop.
3. Web app: `audiolayer web` or double-click `AudioLayer.command` / `AudioLayer.bat`.
4. Claude on the web (claude.ai) cannot reach a program on the user's laptop; use Claude Desktop or
   deploy the server remotely (docs/MCP.md).
Full Indonesian tutorial: docs/TUTORIAL.md in the repo.
