import { useState } from "react";
import { CATEGORY_META, FLAG_TEXT, fmtTime } from "../categories";
import type { Category, Job, Project, Segment, SystemInfo } from "../services/types";
import { CATEGORIES } from "../services/types";

export function ProgressBar({ value, label }: { value: number; label?: string }) {
  return (
    <div className="progress" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className="progress-fill" style={{ width: `${Math.round(value * 100)}%` }} />
      {label && <span className="progress-label">{label}</span>}
    </div>
  );
}

const sev = { error: "sev-error", warning: "sev-warn", info: "sev-info" } as const;

interface LeftProps {
  project: Project;
  system: SystemInfo | null;
  hiddenLanes: Set<Category>;
  toggleLane: (c: Category) => void;
  onAnalyze: (opts: { separate: boolean; transcribe: boolean; denoise: string }) => void;
  onPreset: (preset: string) => void;
  busy: boolean;
}

export function LeftPanel({ project, system, hiddenLanes, toggleLane, onAnalyze, onPreset, busy }: LeftProps) {
  const m = project.media;
  const a = project.analysis;
  const [separate, setSeparate] = useState(true);
  const [transcribe, setTranscribe] = useState(true);
  const [denoise, setDenoise] = useState("auto");
  const counts = Object.fromEntries(CATEGORIES.map((c) => [c, project.segments.filter((s) => s.category === c).length]));
  const dur = m.audio_duration || 1;
  // coverage = union of segment time for the category (segments may overlap)
  const coverage = (c: Category) => {
    const iv = project.segments.filter((s) => s.category === c).map((s) => [s.start, s.end]).sort((x, y) => x[0] - y[0]);
    let total = 0, end = -1;
    for (const [a, b] of iv) { if (b > end) { total += b - Math.max(a, end); end = b; } }
    return Math.min(1, total / dur);
  };
  return (
    <aside className="panel left">
      <section>
        <h3>Media</h3>
        <dl className="kv">
          <dt>File</dt><dd title={m.filename}>{m.filename}</dd>
          {m.has_video && <><dt>Video</dt><dd>{fmtTime(m.video_duration ?? 0, true)} · {m.width}×{m.height} · {m.fps ?? "?"} fps</dd></>}
          <dt>Audio</dt><dd>{m.has_audio ? `${fmtTime(m.audio_duration ?? 0, true)} · ${m.sample_rate ?? "?"} Hz · ${m.channels ?? "?"} ch · ${m.audio_codec}` : "no audio stream"}</dd>
          <dt>Status</dt><dd><span className={`badge st-${project.status}`}>{project.status}</span></dd>
          {a && <><dt>Loudness</dt><dd>{a.loudness.integrated_lufs?.toFixed(1) ?? "–"} LUFS · TP {a.loudness.true_peak_dbtp?.toFixed(1) ?? "–"} dBTP</dd></>}
        </dl>
        {project.error && <p className="error-text">{project.error}</p>}
      </section>

      <section>
        <h3>Categories</h3>
        <ul className="cat-list">
          {CATEGORIES.map((c) => (
            <li key={c}>
              <label title={system?.categories?.[c] ?? ""}>
                <input type="checkbox" checked={!hiddenLanes.has(c)} onChange={() => toggleLane(c)} />
                <span className="dot" style={{ background: CATEGORY_META[c].color }} />
                {CATEGORY_META[c].label}
                <span className="muted small"> {counts[c]} seg · {Math.round(coverage(c) * 100)}%</span>
              </label>
            </li>
          ))}
        </ul>
        <p className="muted small">Checkbox shows/hides the lane. Use the bottom bar to remove a category from what you hear and export.</p>
      </section>

      {a && (
        <section>
          <h3>Quick fixes</h3>
          <div className="preset-grid">
            {Object.entries(system?.presets ?? {}).map(([k, v]) => (
              <button key={k} className="btn ghost sm" title={v.why} disabled={!a.separated && k !== "clear"} onClick={() => onPreset(k)}>{v.label}</button>
            ))}
          </div>
        </section>
      )}

      {a && a.issues.length > 0 && (
        <section>
          <h3>Detected problems</h3>
          <ul className="issues">
            {a.issues.map((i) => (
              <li key={i.id} className={sev[i.severity]}>
                <strong>{i.title}</strong>
                <p>{i.detail}</p>
                {i.fix === "remove_noise" && a.separated && <button className="btn ghost sm" onClick={() => onPreset("remove_noise")}>Remove noise</button>}
                {i.fix === "remove_gibberish" && <button className="btn ghost sm" onClick={() => onPreset("remove_gibberish")}>Mute gibberish</button>}
                {i.fix === "loudness_preset" && <span className="muted small">→ choose a loudness preset when exporting</span>}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h3>Processing</h3>
        <label className="opt"><input type="checkbox" checked={separate} onChange={(e) => setSeparate(e.target.checked)} disabled={!system?.separator} /> Source separation {system?.separator ? <span className="muted small">({system.separator})</span> : <span className="muted small">(not installed)</span>}</label>
        <label className="opt"><input type="checkbox" checked={transcribe} onChange={(e) => setTranscribe(e.target.checked)} disabled={!system?.asr} /> Transcribe dialogue {!system?.asr && <span className="muted small">(not installed)</span>}</label>
        <label className="opt">Noise stem
          <select value={denoise} onChange={(e) => setDenoise(e.target.value)}>
            <option value="auto">auto (only if detected)</option><option value="on">always</option><option value="off">off</option>
          </select>
        </label>
        <button className="btn primary full" disabled={busy} onClick={() => onAnalyze({ separate, transcribe, denoise })}>
          {project.analysis ? "Re-analyze" : "Analyze audio"}
        </button>
        {a && (
          <details className="models">
            <summary>Models &amp; timings</summary>
            <ul>{a.models.map((mm, i) => <li key={i}><strong>{mm.role}:</strong> {mm.name} <span className="muted">({mm.license})</span></li>)}</ul>
            <p className="muted small">{Object.entries(a.timings).map(([k, v]) => `${k} ${v.toFixed(1)}s`).join(" · ")}</p>
          </details>
        )}
        {system?.notes?.map((n, i) => <p key={i} className="muted small">⚠ {n}</p>)}
      </section>
    </aside>
  );
}

interface RightProps {
  segment: Segment | null;
  project: Project;
  onPatch: (patch: Partial<Pick<Segment, "category" | "muted" | "solo" | "start" | "end">>) => void;
  onToggleRemove: (c: Category) => void;
  onSeek: (t: number) => void;
  activeJob: Job | null;
}

export function RightPanel({ segment: s, project, onPatch, onToggleRemove, onSeek, activeJob }: RightProps) {
  const separated = !!project.analysis?.separated;
  return (
    <aside className="panel right">
      <section>
        <h3>Segment</h3>
        {!s && <p className="muted">Select a segment on the timeline.</p>}
        {s && (
          <>
            <div className="seg-head" style={{ borderColor: CATEGORY_META[s.category].color }}>
              <span className="dot" style={{ background: CATEGORY_META[s.category].color }} />
              <strong>{CATEGORY_META[s.category].label}</strong>
              {s.edited && <span className="badge">edited</span>}
              {s.uncertain && <span className="badge warn">uncertain</span>}
            </div>
            <p className="desc">{s.description}</p>
            <dl className="kv">
              <dt>Start</dt><dd><button className="link mono" onClick={() => onSeek(s.start)}>{fmtTime(s.start, true)}</button></dd>
              <dt>End</dt><dd className="mono">{fmtTime(s.end, true)}</dd>
              <dt>Duration</dt><dd className="mono">{s.duration.toFixed(2)} s</dd>
              <dt>Confidence</dt><dd><ProgressBar value={s.confidence} label={`${Math.round(s.confidence * 100)}%`} /></dd>
              {s.intelligibility != null && <><dt>Intelligibility</dt><dd><ProgressBar value={s.intelligibility} label={`${Math.round(s.intelligibility * 100)}%`} /></dd></>}
              <dt>Source</dt><dd>{s.stream === "mix" ? "mix (not separable)" : `${s.stream} stream`}</dd>
              {s.original_category && s.original_category !== s.category && <><dt>Detected as</dt><dd>{CATEGORY_META[s.original_category].label}</dd></>}
            </dl>
            {s.transcript && <blockquote className="transcript">{s.transcript}</blockquote>}
            {s.labels.length > 0 && <p className="muted small">Model labels: {s.labels.map((l) => `${l.label} ${(l.score * 100).toFixed(0)}%`).join(", ")}</p>}
            {s.flags.length > 0 && <ul className="flags">{s.flags.map((f) => <li key={f}>{FLAG_TEXT[f] ?? f}</li>)}</ul>}

            <h4>Category</h4>
            <div className="cat-picker">
              {CATEGORIES.map((c) => (
                <button key={c} className={`chip ${s.category === c ? "on" : ""}`} style={{ ["--c" as string]: CATEGORY_META[c].color }}
                  onClick={() => onPatch({ category: c })} disabled={!!activeJob && activeJob.kind === "analyze"}>{CATEGORY_META[c].label}</button>
              ))}
            </div>
            {(s.category === "bgm" || s.category === "music") && <p className="muted small">BGM = bed under the action; Music = song/performance in the foreground. Fix it here if the role is wrong.</p>}

            <h4>Actions</h4>
            <div className="row gap wrap">
              <button className={`btn ${s.solo ? "warn" : "ghost"}`} disabled={!separated || s.stream === "mix"} onClick={() => onPatch({ solo: !s.solo })} title="Hear only this segment">{s.solo ? "Un-isolate" : "Isolate"}</button>
              <button className={`btn ${s.muted ? "danger" : "ghost"}`} disabled={!separated || s.stream === "mix"} onClick={() => onPatch({ muted: !s.muted })} title="Remove this segment from the mix">{s.muted ? "Restore" : "Remove segment"}</button>
              <button className="btn ghost" disabled={!separated} onClick={() => onToggleRemove(s.category)}>
                {project.mix.removed.includes(s.category) ? `Keep all ${CATEGORY_META[s.category].label}` : `Remove all ${CATEGORY_META[s.category].label}`}
              </button>
            </div>
            <h4>Boundaries</h4>
            <div className="row gap">
              <label className="small">Start <input type="number" step={0.01} min={0} defaultValue={s.start.toFixed(3)} key={`${s.id}-s-${s.start}`} onBlur={(e) => Number(e.target.value) !== s.start && onPatch({ start: Number(e.target.value) })} /></label>
              <label className="small">End <input type="number" step={0.01} min={0} defaultValue={s.end.toFixed(3)} key={`${s.id}-e-${s.end}`} onBlur={(e) => Number(e.target.value) !== s.end && onPatch({ end: Number(e.target.value) })} /></label>
            </div>
          </>
        )}
      </section>
      {project.limitations_for_selection.length > 0 && (
        <section>
          <h3>Limitations</h3>
          <ul className="limits">{project.limitations_for_selection.map((l, i) => <li key={i}>{l}</li>)}</ul>
        </section>
      )}
      {project.analysis && project.analysis.limitations.length > 0 && (
        <section>
          <h3>Analysis notes</h3>
          <ul className="limits">{project.analysis.limitations.map((l, i) => <li key={i}>{l}</li>)}</ul>
        </section>
      )}
    </aside>
  );
}
