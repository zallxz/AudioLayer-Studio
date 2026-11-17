# Using AudioLayer from Claude (MCP)

AudioLayer exposes real processing through the Model Context Protocol using the official
Python SDK (`mcp` v2, `MCPServer`). There are two very different ways to connect.

## Local vs. remote — read this first

| | **Local (stdio)** | **Remote (Streamable HTTP)** |
|---|---|---|
| Who can use it | Claude Desktop / Claude Code on the same computer | Claude apps that support custom remote connectors (incl. claude.ai on the web), if your plan/org allows them |
| How it runs | Claude launches `audiolayer-mcp` as a child process | You deploy `audiolayer-api --remote-mcp` on a server with a public **HTTPS** URL |
| Media access | Reads files from folders you list in `AUDIOLAYER_INPUT_DIRS` | Files are uploaded through a single-use upload URL; no server paths are accepted |
| Outputs | Absolute file paths on your disk | Download URLs (`/api/projects/{id}/files/...`, bearer token required) |
| Auth | None needed (local process) | `Authorization: Bearer <token>` (`AUDIOLAYER_AUTH_TOKENS`); projects are owned by the token's principal |

**A server running on your laptop does not automatically work with Claude on the web.** claude.ai
connects to remote connectors from the internet, so `localhost` is unreachable from it. Use the local
stdio setup with Claude Desktop, or deploy the remote server behind TLS.

## Local setup (Claude Desktop)

**Shortcut:** `audiolayer setup-claude --input-dir <folder>` does steps 1–2 for you (add `--code` for
Claude Code). The manual route:

1. Install AudioLayer (see README) and note the absolute path of the `audiolayer-mcp` executable
   (`which audiolayer-mcp` on macOS/Linux, `where audiolayer-mcp` on Windows).
2. Edit Claude Desktop's config (`Settings → Developer → Edit Config`):
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "audiolayer": {
      "command": "/absolute/path/to/.venv/bin/audiolayer-mcp",
      "env": {
        "AUDIOLAYER_WORKSPACE": "/Users/me/AudioLayer/workspace",
        "AUDIOLAYER_INPUT_DIRS": "/Users/me/Movies/ai-clips",
        "AUDIOLAYER_DEVICE": "auto"
      }
    }
  }
}
```

3. Restart Claude Desktop. Ask: *"Analyze ~/Movies/ai-clips/scene12.mp4 with AudioLayer and remove the
   background music."*

Claude Code: `claude mcp add audiolayer --env AUDIOLAYER_INPUT_DIRS=$HOME/Movies -- audiolayer-mcp`

The web UI and the MCP server can share one workspace: projects created through Claude appear in the
UI's project list (run `audiolayer-api` with the same `AUDIOLAYER_WORKSPACE`).

Testing without Claude: `npx @modelcontextprotocol/inspector audiolayer-mcp` (official MCP Inspector).

## Remote deployment

```bash
export AUDIOLAYER_AUTH_TOKENS="alice:$(openssl rand -hex 24)"
export AUDIOLAYER_PUBLIC_BASE_URL=https://audiolayer.example.com
export AUDIOLAYER_ALLOWED_HOSTS=audiolayer.example.com
audiolayer-api --host 127.0.0.1 --port 8000 --remote-mcp   # behind Caddy/nginx with TLS
```

* MCP endpoint: `https://audiolayer.example.com/mcp/` (stateless Streamable HTTP, JSON responses,
  DNS-rebinding protection with an explicit host allowlist).
* Every MCP and API request must carry the bearer token. The current implementation uses static
  tokens; for a public multi-tenant service put an OAuth 2.1 authorization server in front
  (the SDK's `MCPServer(auth=..., token_verifier=...)` hooks are the place to wire it).
* Upload flow: `create_upload_url` → user runs `curl -F file=@clip.mp4 <url>` → returns `project_id`
  → `analyze_media(project_id=...)`. Upload URLs are single-use and expire after one hour.
* Uploaded media is stored only on your server; nothing is sent to third-party APIs.
* Run `AUDIOLAYER_WORKERS=1` per CPU-bound node. For horizontal scale, swap the in-process
  `JobManager` for a queue (Celery/RQ/Arq) and the workspace for shared storage — the service layer
  already isolates both.

## Tools

| Tool | Purpose | Long-running? |
|---|---|---|
| `analyze_media` | Ingest a file (local path) or project and start detection/separation/classification | returns `job_id` |
| `get_processing_status` | Job progress; lists output files when an export finishes | — |
| `cancel_processing` | Cancel at the next chunk boundary | — |
| `get_analysis` | Media info, loudness, problems, models, coverage, limitations | — |
| `list_audio_segments` | Paged segments with category, times, confidence, flags, transcript | — |
| `update_segment_category` | Relabel / re-time / mute a segment (undoable) | re-render in background |
| `remove_audio_category` | Choose categories to remove, or a preset; returns limitations + preview job | optional `job_id` |
| `separate_audio_category` | Export one category as a full-length stem | `job_id` |
| `export_audio_stems` | Export stems + original + metadata + ZIP | `job_id` |
| `export_cleaned_audio` | Export the cleaned mix (+ loudness preset, + video remux) | `job_id` |
| `get_quality_report` | Creator-focused problem list with fixes | — |
| `list_projects`, `get_system_info` | Discovery | — |
| `create_upload_url` | Remote only: single-use upload link | — |

Presets: `remove_music`, `dialogue_only`, `music_and_effects`, `remove_noise`, `remove_sfx`,
`remove_gibberish`, `clear`.

### Example calls

```jsonc
// 1. start
{"tool": "analyze_media", "arguments": {"file_path": "/Users/me/Movies/ai-clips/scene12.mp4"}}
// → {"project_id": "prj_3f…", "job_id": "job_9a…", "status": "running", "media": {...}}

// 2. poll
{"tool": "get_processing_status", "arguments": {"job_id": "job_9a…"}}
// → {"status": "running", "stage": "separation", "progress": 0.41, "next": "Call get_processing_status again…"}

// 3. inspect
{"tool": "list_audio_segments", "arguments": {"project_id": "prj_3f…", "category": "dialogue"}}

// 4. fix a wrong label
{"tool": "update_segment_category", "arguments": {"project_id": "prj_3f…", "segment_id": "seg_12…", "category": "music"}}

// 5. remove and export
{"tool": "remove_audio_category", "arguments": {"project_id": "prj_3f…", "preset": "remove_music"}}
{"tool": "export_cleaned_audio", "arguments": {"project_id": "prj_3f…", "loudness_preset": "youtube", "include_video": true}}
// poll → {"status": "succeeded", "files": [{"path": ".../scene12_cleaned.wav"}, {"path": ".../scene12_cleaned_video.mp4"}], "limitations": [...]}
```

Errors come back as MCP tool errors with a code and a hint, e.g.
`[forbidden] Path is outside the allowed input folders: /etc/passwd Hint: Allowed folders: /Users/me/Movies/ai-clips`.
