import type { Category, Segment } from "./types";
import { CATEGORIES } from "./types";

/**
 * Sample-accurate stem preview with the Web Audio API.
 *
 * Every category stem (rendered by the backend from the separated streams) plays through its
 * own GainNode. Removing a category, muting a segment or soloing a segment is expressed as gain
 * automation, so what you hear is the same reconstruction the exporter performs. When no
 * stems exist (detection-only analysis) the original audio is played instead.
 */
export type EngineMode = "stems" | "original";

const RAMP = 0.02;

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private buffers = new Map<string, AudioBuffer>();
  private sources: AudioBufferSourceNode[] = [];
  private gains = new Map<string, GainNode>();
  private master: GainNode | null = null;
  private startCtxTime = 0;
  private startOffset = 0;
  private _playing = false;
  private loadToken = 0;
  duration = 0;
  mode: EngineMode = "stems";
  removed = new Set<Category>();
  segments: Segment[] = [];
  onEnded: (() => void) | null = null;

  private ensureCtx(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext({ sampleRate: 48000 });
      this.master = this.ctx.createGain();
      this.master.connect(this.ctx.destination);
    }
    return this.ctx;
  }

  get playing() { return this._playing; }
  get hasStems() { return CATEGORIES.every((c) => this.buffers.has(c)); }
  get loaded() { return this.buffers.has("original") || this.hasStems; }

  async load(urls: Partial<Record<Category | "original", string>>, onProgress?: (f: number) => void) {
    const token = ++this.loadToken;
    const ctx = this.ensureCtx();
    const entries = Object.entries(urls).filter(([, u]) => !!u) as [string, string][];
    let done = 0;
    const decoded = await Promise.all(entries.map(async ([k, url]) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Failed to load ${k} (${res.status})`);
      const buf = await ctx.decodeAudioData(await res.arrayBuffer());
      onProgress?.(++done / entries.length);
      return [k, buf] as const;
    }));
    if (token !== this.loadToken) return; // superseded by a newer load
    const wasPlaying = this._playing;
    const pos = this.position;
    this.stopSources();
    for (const [k, b] of decoded) this.buffers.set(k, b);
    this.duration = Math.max(...[...this.buffers.values()].map((b) => b.duration), 0);
    if (wasPlaying) this.play(pos);
  }

  clear() {
    this.stop();
    this.buffers.clear();
    this.duration = 0;
  }

  setVolume(v: number) {
    const ctx = this.ensureCtx();
    this.master!.gain.setTargetAtTime(v, ctx.currentTime, 0.01);
  }

  get position(): number {
    if (!this.ctx || !this._playing) return this.startOffset;
    return Math.max(0, Math.min(this.duration, this.startOffset + (this.ctx.currentTime - this.startCtxTime)));
  }

  play(from?: number) {
    const ctx = this.ensureCtx();
    if (ctx.state === "suspended") void ctx.resume();
    this.stopSources();
    const offset = Math.max(0, Math.min(from ?? this.startOffset, this.duration - 0.01));
    this.startOffset = offset;
    this.startCtxTime = ctx.currentTime + 0.05;
    const keys = this.mode === "stems" && this.hasStems ? CATEGORIES : ["original"];
    this.gains.clear();
    for (const k of keys) {
      const buf = this.buffers.get(k);
      if (!buf) continue;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      const g = ctx.createGain();
      src.connect(g).connect(this.master!);
      src.start(this.startCtxTime, offset);
      this.sources.push(src);
      this.gains.set(k, g);
    }
    const first = this.sources[0];
    if (first) first.onended = () => {
      if (this._playing && this.position >= this.duration - 0.05) {
        this._playing = false;
        this.startOffset = 0;
        this.onEnded?.();
      }
    };
    this._playing = true;
    this.applyAutomation();
  }

  pause() {
    if (!this._playing) return;
    this.startOffset = this.position;
    this.stopSources();
    this._playing = false;
  }

  stop() {
    this.stopSources();
    this._playing = false;
    this.startOffset = 0;
  }

  seek(t: number) {
    if (this._playing) this.play(t);
    else this.startOffset = Math.max(0, Math.min(t, this.duration));
  }

  /** Re-apply category removal + segment mute/solo automation from "now" onwards. */
  applyAutomation() {
    if (!this.ctx || !this._playing) return;
    const now = Math.max(this.ctx.currentTime, this.startCtxTime);
    const t0 = this.startOffset + (now - this.startCtxTime);
    const anySolo = this.segments.some((s) => s.solo);
    for (const [k, g] of this.gains) {
      g.gain.cancelScheduledValues(0);
      if (k === "original") { g.gain.setValueAtTime(1, now); continue; }
      const cat = k as Category;
      const base = anySolo ? 0 : this.removed.has(cat) ? 0 : 1;
      g.gain.setValueAtTime(base, now);
      // Build an ordered list of [start, end, value] overrides for this category.
      const overrides: [number, number, number][] = [];
      for (const s of this.segments) {
        if (s.category !== cat || s.end < t0) continue;
        if (anySolo && s.solo) overrides.push([s.start, s.end, 1]);
        else if (!anySolo && base === 1 && s.muted) overrides.push([s.start, s.end, 0]);
      }
      overrides.sort((a, b) => a[0] - b[0]);
      for (const [a, b, v] of overrides) {
        const ta = now + Math.max(0, a - t0);
        const tb = now + Math.max(0, b - t0);
        if (a <= t0) g.gain.setValueAtTime(v, now);
        else { g.gain.setValueAtTime(base, ta); g.gain.linearRampToValueAtTime(v, ta + RAMP); }
        g.gain.setValueAtTime(v, tb);
        g.gain.linearRampToValueAtTime(base, tb + RAMP);
      }
    }
  }

  private stopSources() {
    for (const s of this.sources) {
      s.onended = null;
      try { s.stop(); } catch { /* already stopped */ }
      s.disconnect();
    }
    this.sources = [];
  }
}
