import { useState } from "react";
import { CATEGORY_META } from "../categories";
import { api, fileUrl, pollJob } from "../services/api";
import type { Job, Project } from "../services/types";
import { CATEGORIES } from "../services/types";
import { ProgressBar } from "./Panels";

const ITEMS: { key: string; label: string; needsStems?: boolean; needsVideo?: boolean }[] = [
  { key: "mix", label: "Cleaned mix" },
  ...CATEGORIES.map((c) => ({ key: c, label: `${CATEGORY_META[c].label} stem`, needsStems: true })),
  { key: "original", label: "Original extracted audio" },
  { key: "video", label: "Video with cleaned audio", needsVideo: true },
  { key: "metadata", label: "Metadata (JSON, CSV, SRT, Audacity labels)" },
];

export function ExportDialog({ project, onClose }: { project: Project; onClose: () => void }) {
  const stems = project.stems_available;
  const [items, setItems] = useState<Set<string>>(new Set(["mix", ...(stems ? CATEGORIES : []), "original", "metadata"]));
  const [format, setFormat] = useState("wav");
  const [loud, setLoud] = useState("none");
  const [fade, setFade] = useState(5);
  const [zip, setZip] = useState(true);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const toggle = (k: string) => setItems((s) => { const n = new Set(s); if (n.has(k)) n.delete(k); else n.add(k); return n; });

  const start = async () => {
    setError(null);
    try {
      const j = await api.export(project.id, { items: [...items], format, loudness_preset: loud, edge_fade_ms: fade, zip });
      setJob(j);
      await pollJob(j.id, setJob);
    } catch (e: any) { setError(e.message ?? String(e)); }
  };

  const res = job?.status === "succeeded" ? job.result : null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Export">
        <h2>Export</h2>
        <p className="muted small">Removing: {project.mix.removed.length ? project.mix.removed.map((c) => CATEGORY_META[c].label).join(", ") : "nothing"} · muted segments: {project.segments.filter((s) => s.muted).length}. All files are full length and start at 0 s, so they stay in sync in any editor.</p>
        <div className="export-grid">
          <div>
            <h4>Files</h4>
            {ITEMS.map((it) => {
              const disabled = (it.needsStems && !stems) || (it.needsVideo && !project.media.has_video);
              return (
                <label key={it.key} className={`opt ${disabled ? "disabled" : ""}`}>
                  <input type="checkbox" disabled={disabled} checked={items.has(it.key) && !disabled} onChange={() => toggle(it.key)} /> {it.label}
                </label>
              );
            })}
          </div>
          <div>
            <h4>Format</h4>
            <select value={format} onChange={(e) => setFormat(e.target.value)}>
              <option value="wav">WAV · 48 kHz · 24-bit</option>
              <option value="flac">FLAC · 48 kHz</option>
              <option value="mp3">MP3 · 320 kbps</option>
            </select>
            <h4>Loudness (cleaned mix)</h4>
            <select value={loud} onChange={(e) => setLoud(e.target.value)}>
              <option value="none">Keep original level</option>
              <option value="youtube">YouTube −14 LUFS / −1 dBTP</option>
              <option value="tiktok">TikTok / Reels / Shorts −14 LUFS</option>
              <option value="podcast">Podcast −16 LUFS</option>
              <option value="broadcast">Broadcast EBU R128 −23 LUFS</option>
            </select>
            <h4>Edge fades</h4>
            <label className="opt"><input type="number" min={0} max={500} value={fade} onChange={(e) => setFade(Number(e.target.value))} /> ms (prevents clicks when joining clips)</label>
            <label className="opt"><input type="checkbox" checked={zip} onChange={(e) => setZip(e.target.checked)} /> ZIP package for your editor</label>
          </div>
        </div>
        {job && !res && <ProgressBar value={job.progress} label={`${job.message} ${Math.round(job.progress * 100)}%`} />}
        {job?.status === "failed" && <p className="error-text">{job.error}</p>}
        {error && <p className="error-text">{error}</p>}
        {res && (
          <div className="export-results">
            {res.zip && <a className="btn primary" href={fileUrl(project.id, res.zip, true)}>Download ZIP</a>}
            <ul>{(res.files as string[]).map((f) => <li key={f}><a href={fileUrl(project.id, `${res.dir}/${f}`, true)}>{f}</a></li>)}</ul>
            {(res.skipped as { item: string; reason: string }[]).map((s, i) => <p key={i} className="muted small">Skipped {s.item}: {s.reason}</p>)}
            {(res.limitations as string[] | undefined)?.map((l, i) => <p key={i} className="muted small">⚠ {l}</p>)}
          </div>
        )}
        <div className="row gap end">
          <button className="btn ghost" onClick={onClose}>Close</button>
          <button className="btn primary" disabled={!!job && !["succeeded", "failed", "cancelled"].includes(job.status) || items.size === 0} onClick={start}>Export</button>
        </div>
      </div>
    </div>
  );
}
