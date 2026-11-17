import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CATEGORY_META, fmtTime } from "./categories";
import { api, ApiError, fileUrl, getToken, mediaUrl, pollJob, setToken } from "./services/api";
import { AudioEngine } from "./services/audioEngine";
import type { Category, Job, Peaks, Project, ProjectSummary, Segment, SystemInfo } from "./services/types";
import { CATEGORIES } from "./services/types";
import { Timeline } from "./timeline/Timeline";
import { ExportDialog } from "./workspace/ExportDialog";
import { LeftPanel, ProgressBar, RightPanel } from "./workspace/Panels";

const ACCEPT = ".mp4,.mov,.mkv,.webm,.wav,.mp3,.aac,.m4a,.flac";

export default function App() {
  const engine = useMemo(() => new AudioEngine(), []);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [peaks, setPeaks] = useState<Peaks | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [upload, setUpload] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hiddenLanes, setHiddenLanes] = useState<Set<Category>>(new Set());
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(0.9);
  const [abOriginal, setAbOriginal] = useState(false);
  const [loading, setLoading] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showExport, setShowExport] = useState(false);
  const [needToken, setNeedToken] = useState(false);
  const [, force] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const loadedStems = useRef<string>("");

  const report = (e: unknown) => {
    if (e instanceof ApiError && e.status === 401) { setNeedToken(true); return; }
    setError(e instanceof ApiError ? `${e.message}${e.hint ? ` — ${e.hint}` : ""}` : String((e as Error)?.message ?? e));
  };

  const refreshProjects = useCallback(() => api.projects().then(setProjects).catch(report), []);
  useEffect(() => { api.system().then(setSystem).catch(report); refreshProjects(); }, [refreshProjects]);

  const openProject = useCallback(async (id: string, keepPosition = false) => {
    try {
      const pr = await api.project(id);
      setProject(pr);
      if (!keepPosition) { engine.clear(); loadedStems.current = ""; setPlaying(false); setSelectedId(null); }
      api.peaks(id).then(setPeaks).catch(() => setPeaks(null));
      const running = pr.active_jobs.find((j) => j.kind === "analyze");
      if (running) track(running);
      history.replaceState(null, "", `#${id}`);
    } catch (e) { report(e); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine]);

  useEffect(() => {
    const id = location.hash.slice(1);
    if (id) void openProject(id);
  }, [openProject]);

  // (Re)load audio when stems change
  useEffect(() => {
    if (!project) return;
    const key = `${project.id}:${project.stems_available ? project.stems_version : "orig"}`;
    if (loadedStems.current === key) return;
    loadedStems.current = key;
    const urls: Partial<Record<Category | "original", string>> = { original: fileUrl(project.id, "audio/master.wav") };
    if (project.stems_available) for (const c of CATEGORIES) urls[c] = `${fileUrl(project.id, `stems/${c}.flac`)}${getToken() ? "&" : "?"}v=${project.stems_version}`;
    setLoading(0);
    engine.load(urls, setLoading).then(() => { setLoading(null); force((n) => n + 1); }).catch((e) => { setLoading(null); report(e); });
  }, [project, engine]);

  // keep engine state in sync with project state
  useEffect(() => {
    if (!project) return;
    engine.removed = new Set(project.mix.removed);
    engine.segments = project.segments;
    const mode = abOriginal ? "original" : "stems";
    if (engine.mode !== mode) {
      engine.mode = mode;
      if (engine.playing) engine.play(engine.position);
    }
    engine.applyAutomation();
  }, [project, engine, abOriginal]);

  useEffect(() => { engine.setVolume(volume); }, [volume, engine]);
  useEffect(() => { engine.onEnded = () => { setPlaying(false); videoRef.current?.pause(); }; }, [engine]);

  // video follows the audio clock
  useEffect(() => {
    const id = setInterval(() => {
      const v = videoRef.current;
      if (!v || !engine.playing) return;
      if (Math.abs(v.currentTime - engine.position) > 0.12) v.currentTime = engine.position;
    }, 250);
    return () => clearInterval(id);
  }, [engine]);

  const track = useCallback(async (j: Job) => {
    setJob(j);
    try {
      const done = await pollJob(j.id, setJob);
      if (done.status === "failed") setError(done.error ?? "Job failed");
      await openProject(done.project_id, done.kind !== "analyze");
      if (done.kind === "analyze") refreshProjects();
    } catch (e) { report(e); } finally { setTimeout(() => setJob((cur) => (cur && cur.id === j.id ? null : cur)), 1500); }
  }, [openProject, refreshProjects]);

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setError(null);
    setUpload(0);
    try {
      const res = await api.upload(f, true, setUpload);
      setUpload(null);
      await openProject(res.project.id);
      refreshProjects();
      if (res.job) void track(res.job);
    } catch (e) { setUpload(null); report(e); }
  };

  const play = () => {
    if (!engine.loaded) return;
    if (engine.playing) { engine.pause(); videoRef.current?.pause(); setPlaying(false); }
    else {
      engine.play();
      const v = videoRef.current;
      if (v) { v.currentTime = engine.position; void v.play().catch(() => undefined); }
      setPlaying(true);
    }
  };
  const seek = (t: number) => {
    engine.seek(t);
    if (videoRef.current) videoRef.current.currentTime = t;
    force((n) => n + 1);
  };

  const patchSegment = async (patch: Partial<Pick<Segment, "category" | "muted" | "solo" | "start" | "end">>) => {
    if (!project || !selectedId) return;
    try {
      const res = await api.patchSegment(project.id, selectedId, patch);
      setProject((p) => p && { ...p, can_undo: true, segments: p.segments.map((s) => (s.id === selectedId ? res.segment : s)) });
      if (res.render_job) void track(res.render_job);
      else void openProject(project.id, true);
    } catch (e) { report(e); }
  };

  const setRemoved = async (removed: Category[]) => {
    if (!project) return;
    try {
      const res = await api.setMix(project.id, { removed });
      setProject((p) => p && { ...p, mix: { removed: res.removed }, limitations_for_selection: res.limitations, can_undo: true });
    } catch (e) { report(e); }
  };
  const toggleRemove = (c: Category) => project && setRemoved(project.mix.removed.includes(c) ? project.mix.removed.filter((x) => x !== c) : [...project.mix.removed, c]);
  const applyPreset = async (preset: string) => {
    if (!project) return;
    try {
      const res = await api.setMix(project.id, { preset });
      setProject((p) => p && { ...p, mix: { removed: res.removed }, segments: res.segments, limitations_for_selection: res.limitations, can_undo: true });
    } catch (e) { report(e); }
  };
  const undoRedo = async (redo: boolean) => {
    if (!project) return;
    try { await (redo ? api.redo(project.id) : api.undo(project.id)); await openProject(project.id, true); } catch (e) { report(e); }
  };
  const analyze = async (opts: { separate: boolean; transcribe: boolean; denoise: string }) => {
    if (!project) return;
    try { const j = await api.analyze(project.id, opts); setProject({ ...project, status: "analyzing" }); void track(j); } catch (e) { report(e); }
  };

  // keyboard: space = play/pause
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT" || (e.target as HTMLElement)?.tagName === "SELECT") return;
      if (e.code === "Space") { e.preventDefault(); play(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // position ticker for the transport display
  const [pos, setPos] = useState(0);
  useEffect(() => { const id = setInterval(() => setPos(engine.position), 100); return () => clearInterval(id); }, [engine]);

  const selected = project?.segments.find((s) => s.id === selectedId) ?? null;
  const analyzing = job && ["queued", "running"].includes(job.status) && job.kind === "analyze";
  const duration = engine.duration || project?.media.audio_duration || 0;

  return (
    <div className="app" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); void onFile(e.dataTransfer.files[0]); }}>
      <header className="topbar">
        <div className="brand"><span className="logo">◢◤</span> AudioLayer <span className="muted">Studio</span></div>
        <select className="project-select" value={project?.id ?? ""} onChange={(e) => e.target.value && openProject(e.target.value)} aria-label="Project">
          <option value="">{projects.length ? "Open project…" : "No projects yet"}</option>
          {projects.map((p) => <option key={p.id} value={p.id}>{p.name} · {p.status}</option>)}
        </select>
        <button className="btn primary" onClick={() => fileInput.current?.click()}>Upload media</button>
        <input ref={fileInput} type="file" accept={ACCEPT} hidden onChange={(e) => { void onFile(e.target.files?.[0]); e.target.value = ""; }} />
        <div className="status">
          {upload != null && <ProgressBar value={upload} label={`Uploading ${Math.round(upload * 100)}%`} />}
          {job && upload == null && (
            <div className="row gap">
              <ProgressBar value={job.progress} label={`${job.status === "running" || job.status === "queued" ? job.message || job.stage : job.status} ${Math.round(job.progress * 100)}%`} />
              {["queued", "running"].includes(job.status) && <button className="btn ghost sm" onClick={() => api.cancel(job.id).catch(report)}>Cancel</button>}
            </div>
          )}
          {loading != null && !job && <ProgressBar value={loading} label="Loading stems…" />}
        </div>
        <button className="btn ghost" disabled={!project?.can_undo} onClick={() => undoRedo(false)} title="Undo">↶</button>
        <button className="btn ghost" disabled={!project?.can_redo} onClick={() => undoRedo(true)} title="Redo">↷</button>
        <button className="btn accent" disabled={!project || project.status !== "analyzed"} onClick={() => setShowExport(true)}>Export</button>
      </header>

      {error && <div className="toast" role="alert"><span>{error}</span><button className="btn ghost sm" onClick={() => setError(null)}>×</button></div>}
      {needToken && (
        <div className="toast"><span>This server requires an access token.</span>
          <input placeholder="token" onKeyDown={(e) => { if (e.key === "Enter") { setToken((e.target as HTMLInputElement).value); setNeedToken(false); location.reload(); } }} />
        </div>
      )}

      {!project ? (
        <main className="empty" onClick={() => fileInput.current?.click()}>
          <div className="drop">
            <h1>Clean up the audio of your AI-generated video</h1>
            <p>Drop an MP4, MOV, MKV, WAV, MP3, AAC or FLAC file here — or click to choose.</p>
            <p className="muted">AudioLayer finds dialogue, background music, sound effects, foreground music and noise on a timeline, separates them, and lets you remove what you don't want before exporting stems for your editor.</p>
            {system && <p className="muted small">Models: {system.classifier ?? "–"} · {system.separator ?? "no separator"} · {system.asr ?? "no ASR"} · device {system.device}</p>}
          </div>
        </main>
      ) : (
        <div className="workspace">
          <LeftPanel project={project} system={system} hiddenLanes={hiddenLanes}
            toggleLane={(c) => setHiddenLanes((s) => { const n = new Set(s); if (n.has(c)) n.delete(c); else n.add(c); return n; })}
            onAnalyze={analyze} onPreset={applyPreset} busy={!!analyzing} />
          <main className="center">
            <div className="viewer">
              {project.media.has_video
                ? <video ref={videoRef} src={mediaUrl(project.id)} muted playsInline preload="auto" />
                : <div className="no-video">Audio-only file</div>}
            </div>
            {project.status === "analyzed" || peaks ? (
              <Timeline peaks={peaks} segments={project.segments} duration={duration} getPosition={() => engine.position}
                selectedId={selectedId} removed={project.mix.removed} hiddenLanes={hiddenLanes}
                onSelect={setSelectedId} onSeek={seek} />
            ) : <div className="placeholder">Analyze the audio to see the timeline.</div>}
            {project.status === "analyzing" && <p className="muted center-text">Analysis is running — separation is the slow part (roughly real-time on a laptop CPU, much faster on a GPU).</p>}
          </main>
          <RightPanel segment={selected} project={project} onPatch={patchSegment} onToggleRemove={toggleRemove} onSeek={seek} activeJob={job} />
        </div>
      )}

      {project && (
        <footer className="transport">
          <button className="btn play" onClick={play} disabled={!engine.loaded} aria-label={playing ? "Pause" : "Play"}>{playing ? "❚❚" : "▶"}</button>
          <span className="mono time">{fmtTime(pos, true)} / {fmtTime(duration, true)}</span>
          <input type="range" min={0} max={duration || 1} step={0.01} value={pos} onChange={(e) => seek(Number(e.target.value))} className="seek" aria-label="Position" />
          <label className="vol">Vol <input type="range" min={0} max={1.5} step={0.01} value={volume} onChange={(e) => setVolume(Number(e.target.value))} /></label>
          <div className="toggles" aria-label="Category preview toggles">
            {CATEGORIES.map((c) => {
              const on = !project.mix.removed.includes(c);
              return (
                <button key={c} className={`chip ${on ? "on" : "off"}`} style={{ ["--c" as string]: CATEGORY_META[c].color }}
                  disabled={!project.stems_available || abOriginal} onClick={() => toggleRemove(c)}
                  title={project.stems_available ? (on ? `Remove ${CATEGORY_META[c].label}` : `Restore ${CATEGORY_META[c].label}`) : "Needs separation"}>
                  {CATEGORY_META[c].label}
                </button>
              );
            })}
          </div>
          <button className={`btn ${abOriginal ? "warn" : "ghost"}`} onClick={() => setAbOriginal((v) => !v)} disabled={!project.stems_available} title="Compare with the untouched original">A/B original</button>
        </footer>
      )}
      {showExport && project && <ExportDialog project={project} onClose={() => setShowExport(false)} />}
    </div>
  );
}
