import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CATEGORY_META, fmtTime } from "../categories";
import type { Category, Peaks, Segment } from "../services/types";
import { CATEGORIES } from "../services/types";

interface Props {
  peaks: Peaks | null;
  segments: Segment[];
  duration: number;
  getPosition: () => number;
  selectedId: string | null;
  removed: Category[];
  hiddenLanes: Set<Category>;
  onSelect: (id: string | null) => void;
  onSeek: (t: number) => void;
}

const RULER_H = 22;
const WAVE_H = 96;
const LANE_H = 30;
const LABEL_W = 64;

export function Timeline(p: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(800);
  const [zoom, setZoom] = useState(1); // 1 = fit to width
  const [scroll, setScroll] = useState(0); // seconds at left edge
  const lanes = useMemo(() => CATEGORIES.filter((c) => !p.hiddenLanes.has(c)), [p.hiddenLanes]);
  const height = RULER_H + WAVE_H + lanes.length * LANE_H + 8;
  const dur = Math.max(p.duration, 0.001);
  const visible = dur / zoom;
  const pxPerSec = (width - LABEL_W) / visible;
  const x = useCallback((t: number) => LABEL_W + (t - scroll) * pxPerSec, [scroll, pxPerSec]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => { setScroll((s) => Math.max(0, Math.min(s, dur - visible))); }, [zoom, dur, visible]);

  // hit-testing data for clicks
  const boxes = useRef<{ id: string; x0: number; x1: number; y0: number; y1: number }[]>([]);

  const draw = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(width * dpr) || cv.height !== Math.round(height * dpr)) {
      cv.width = Math.round(width * dpr);
      cv.height = Math.round(height * dpr);
    }
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, height);
    g.font = "11px ui-sans-serif, system-ui, sans-serif";

    // ruler
    g.fillStyle = "#11151b";
    g.fillRect(0, 0, width, RULER_H);
    const step = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300].find((s) => s * pxPerSec >= 70) ?? 600;
    g.fillStyle = "#6b7785";
    g.strokeStyle = "#252c35";
    for (let t = Math.ceil(scroll / step) * step; t <= scroll + visible + 1e-9; t += step) {
      const xx = x(t);
      g.beginPath(); g.moveTo(xx + 0.5, RULER_H - 6); g.lineTo(xx + 0.5, height); g.stroke();
      g.fillText(fmtTime(t, step < 1), xx + 3, 14);
    }

    // waveform
    const wy = RULER_H, mid = wy + WAVE_H / 2;
    g.fillStyle = "#0d1015";
    g.fillRect(LABEL_W, wy, width - LABEL_W, WAVE_H);
    if (p.peaks && p.peaks.max.length) {
      const pps = p.peaks.points_per_second;
      g.fillStyle = "#5d6b7a";
      for (let px = LABEL_W; px < width; px++) {
        const t0 = scroll + (px - LABEL_W) / pxPerSec, t1 = t0 + 1 / pxPerSec;
        const i0 = Math.floor(t0 * pps), i1 = Math.max(i0 + 1, Math.ceil(t1 * pps));
        let lo = 0, hi = 0;
        for (let i = i0; i < i1 && i < p.peaks.max.length; i++) {
          if (i < 0) continue;
          if (p.peaks.min[i] < lo) lo = p.peaks.min[i];
          if (p.peaks.max[i] > hi) hi = p.peaks.max[i];
        }
        const y0 = mid - hi * (WAVE_H / 2 - 4), y1 = mid - lo * (WAVE_H / 2 - 4);
        g.fillRect(px, y0, 1, Math.max(1, y1 - y0));
      }
    }
    g.fillStyle = "#8a96a3";
    g.fillText("Mix", 10, wy + 16);

    // lanes
    const hit: typeof boxes.current = [];
    lanes.forEach((cat, li) => {
      const y = RULER_H + WAVE_H + li * LANE_H + 4;
      const meta = CATEGORY_META[cat];
      const off = p.removed.includes(cat);
      g.fillStyle = li % 2 ? "#12161c" : "#141920";
      g.fillRect(0, y, width, LANE_H);
      g.fillStyle = off ? "#4a525c" : meta.color;
      g.fillText(meta.label, 10, y + 19);
      if (off) { g.fillStyle = "#4a525c"; g.fillText("off", 46, y + 19); }
      for (const s of p.segments) {
        if (s.category !== cat) continue;
        const x0 = Math.max(LABEL_W, x(s.start)), x1 = Math.min(width, x(s.end));
        if (x1 < LABEL_W || x0 > width || x1 - x0 < 0.5) continue;
        const sel = s.id === p.selectedId;
        g.globalAlpha = off || s.muted ? 0.22 : 0.35 + 0.6 * s.confidence;
        g.fillStyle = meta.color;
        g.fillRect(x0, y + 4, Math.max(2, x1 - x0), LANE_H - 8);
        g.globalAlpha = 1;
        if (s.uncertain) {
          g.save();
          g.beginPath(); g.rect(x0, y + 4, Math.max(2, x1 - x0), LANE_H - 8); g.clip();
          g.strokeStyle = "rgba(0,0,0,0.35)";
          for (let k = x0 - LANE_H; k < x1; k += 7) { g.beginPath(); g.moveTo(k, y + LANE_H - 4); g.lineTo(k + LANE_H, y + 4 - 0); g.stroke(); }
          g.restore();
        }
        if (s.muted) {
          g.strokeStyle = "#ff5b5b"; g.beginPath(); g.moveTo(x0, y + LANE_H / 2); g.lineTo(x1, y + LANE_H / 2); g.stroke();
        }
        if (s.solo) { g.fillStyle = "#ffe066"; g.fillRect(x0, y + 4, 3, LANE_H - 8); }
        if (sel) { g.strokeStyle = "#ffffff"; g.lineWidth = 2; g.strokeRect(x0 + 1, y + 5, Math.max(2, x1 - x0) - 2, LANE_H - 10); g.lineWidth = 1; }
        if (s.flags.includes("unintelligible")) { g.fillStyle = "#ff5b5b"; g.fillText("?!", Math.min(x1 - 14, x0 + 4), y + 19); }
        if (x1 - x0 > 60) {
          g.fillStyle = "#0b0e12";
          const label = s.transcript ? `“${s.transcript}”` : s.description;
          g.save(); g.beginPath(); g.rect(x0, y, x1 - x0 - 4, LANE_H); g.clip();
          g.fillText(label, x0 + (s.flags.includes("unintelligible") ? 22 : 6), y + 19);
          g.restore();
        }
        hit.push({ id: s.id, x0, x1, y0: y + 4, y1: y + LANE_H - 4 });
      }
    });
    boxes.current = hit;

    // playhead
    const ph = x(p.getPosition());
    if (ph >= LABEL_W && ph <= width) {
      g.strokeStyle = "#ffe066"; g.beginPath(); g.moveTo(ph + 0.5, 0); g.lineTo(ph + 0.5, height); g.stroke();
    }
  }, [width, height, p, lanes, scroll, visible, pxPerSec, x]);

  useEffect(() => {
    let raf = 0;
    const loop = () => { draw(); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [draw]);

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const cx = e.clientX - r.left, cy = e.clientY - r.top;
    if (cx < LABEL_W) return;
    const hitBox = [...boxes.current].reverse().find((b) => cx >= b.x0 && cx <= b.x1 && cy >= b.y0 && cy <= b.y1);
    if (hitBox && cy > RULER_H + WAVE_H) { p.onSelect(hitBox.id); return; }
    p.onSeek(Math.max(0, Math.min(dur, scroll + (cx - LABEL_W) / pxPerSec)));
  };

  const onWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      setZoom((z) => Math.max(1, Math.min(200, z * (e.deltaY < 0 ? 1.2 : 1 / 1.2))));
    } else if (zoom > 1) {
      setScroll((s) => Math.max(0, Math.min(dur - visible, s + (e.deltaY + e.deltaX) / pxPerSec)));
    }
  };

  // follow playhead when zoomed in
  useEffect(() => {
    const id = setInterval(() => {
      const t = p.getPosition();
      if (zoom > 1 && (t < scroll || t > scroll + visible * 0.95)) setScroll(Math.max(0, Math.min(dur - visible, t - visible * 0.1)));
    }, 250);
    return () => clearInterval(id);
  }, [p, zoom, scroll, visible, dur]);

  return (
    <div className="timeline" ref={wrapRef}>
      <div className="timeline-tools">
        <span className="muted small">Click a segment to inspect · click empty space to seek · Ctrl/⌘+wheel to zoom</span>
        <div className="row gap">
          <button className="btn ghost sm" onClick={() => setZoom((z) => Math.max(1, z / 1.5))} aria-label="Zoom out">−</button>
          <span className="small mono">{zoom.toFixed(1)}×</span>
          <button className="btn ghost sm" onClick={() => setZoom((z) => Math.min(200, z * 1.5))} aria-label="Zoom in">+</button>
          <button className="btn ghost sm" onClick={() => { setZoom(1); setScroll(0); }}>Fit</button>
        </div>
      </div>
      <canvas ref={canvasRef} style={{ width: "100%", height }} onClick={onClick} onWheel={onWheel} />
      {zoom > 1 && (
        <input className="scrollbar" type="range" min={0} max={Math.max(0, dur - visible)} step={0.01} value={scroll}
          onChange={(e) => setScroll(Number(e.target.value))} aria-label="Scroll timeline" />
      )}
    </div>
  );
}
