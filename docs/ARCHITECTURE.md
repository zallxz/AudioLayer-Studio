# Architecture

```
            ┌──────────── Web UI (React + TS + Vite) ────────────┐
            │ timeline/waveform · inspector · WebAudio stem mixer │
            └──────────────────────┬──────────────────────────────┘
                                   │ REST (+ Range requests for media)
 Claude Desktop ── stdio ──┐  ┌────▼─────────────── FastAPI app ───────────────┐
                           │  │ /api/*   auth (bearer) · uploads · downloads    │
 Claude (remote) ─ HTTPS ──┼──► /mcp/    MCP Streamable HTTP (stateless)        │
                           │  └────┬────────────────────────────────────────────┘
               ┌───────────▼───────▼──────────┐
               │ AudioLayerService (façade)   │  ownership checks, presets, undo/redo,
               │                              │  limitations, file resolution
               └──────┬──────────────┬────────┘
                      │              │
            JobManager (threads,     ProjectStore (workspace/projects/<id>/…,
            progress, cancel,        atomic JSON, per-project locks)
            persisted state)
                      │
      ┌───────────────▼──────────────────────────────────────────────────────────┐
      │ pipeline                                                                 │
      │  ingest: ffprobe → copy original (read-only) → ffmpeg → master.wav 48k   │
      │  analyze:                                                                │
      │   1 quality (DSP): silence, clipping, hum, hiss, rumble, bandwidth, …    │
      │   2 Separator adapter → vocals.wav + accompaniment.wav (exact complement)│
      │   3 Classifier adapter on each → framewise category probabilities        │
      │   4 routing → streams: voice, music, fx, noise (sum == master)           │
      │   5 segmenter → segments (overlap-aware, confidence, uncertainty, roles) │
      │   6 ASR adapter → transcripts, intelligibility, gibberish flags          │
      │   7 mixer → category stems (preview FLAC), peaks                         │
      │  export: mixer (removal/mute/solo envelopes) → loudnorm → encode → ZIP   │
      └──────────────────────────────────────────────────────────────────────────┘
```

## Key design decisions

**Streams vs. categories.** Models produce *streams* (voice / music / fx / noise). *Categories*
(dialogue, bgm, sfx, music, noise) are labels on time segments of those streams. A category stem is
`Σ stream × routing-envelope`, so relabelling a segment (e.g. BGM → Music) re-routes audio instantly
without re-running a model, and every edit is undoable.

**Exact reconstruction.** `voice + music + fx + noise == master` (checked in tests to float precision).
Removing a category subtracts exactly what the separator attributed to it; with nothing removed the
export is bit-for-bit the original audio. This is what makes removal honest and non-destructive.

**Overlap.** Categories are segmented independently from per-stream probabilities, so dialogue over
music over an impact shows as three overlapping segments.

**Uncertainty is data.** Low-confidence segments are kept and flagged (`uncertain`, `role_uncertain`,
`overlaps_music_split_estimated`, `not_removable`, `heuristic_classifier`), and every analysis and
export returns a `limitations` list.

**Adapters.** `Classifier`, `Separator` and ASR are small protocols (`classification/base.py`,
`separation/base.py`). `core/components.py` picks implementations from config with fallbacks
(PANNs → heuristic, Demucs → detection-only, Whisper → none).

**Streaming I/O.** All audio processing reads/writes in blocks (`audio/io.py`); separation uses
30 s chunks with 2 s cross-faded overlap; memory does not grow with file length (except Whisper,
which holds 16 kHz mono ≈ 115 MB for 30 min).

**Sync.** The master is decoded with `aresample=async=1:first_pts=0`, so t=0 is the container start.
Every stem/mix is exactly as long as the master. Video export copies the video stream and attaches
new audio, so there is no re-encode drift.

**Security.** Uploads are size-limited and extension/ffprobe-validated; filenames are sanitised;
project/job ids are validated; file downloads are restricted to export/render/stem paths inside the
project; local MCP file access is limited to `AUDIOLAYER_INPUT_DIRS` (symlinks resolved); remote MCP
requires bearer tokens, has DNS-rebinding protection, and owners can only see their own projects
(other ids return 404).

## Repository layout

```
backend/audiolayer/
  api/            FastAPI app, auth
  core/           schemas, store, service façade, pipeline, components, errors, path validation
  audio/          ffmpeg wrappers, streaming I/O, quality analyzers, waveform peaks
  classification/ classifier protocol + AudioSet mapping, PANNs, heuristic, segmenter, ASR
  separation/     separator protocol, Demucs, denoise chain, routing, mixer
  export/         exporter, reports (JSON/CSV/SRT/Audacity labels)
  models/         vendored PANNs architecture, model registry/downloader, AudioSet label list
  workers/        background job manager
  mcp_server/     MCP server (named so it never shadows the `mcp` SDK package)
frontend/src/     services (API client, WebAudio engine), timeline, workspace panels
tests/            synthetic-audio tests; `-m models` real-model integration test
scripts/          download_models, make_demo_media (ground-truth scene), evaluate (SI-SDR)
```
