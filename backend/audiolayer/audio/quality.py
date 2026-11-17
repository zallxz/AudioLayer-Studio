"""Signal-level analyzers for the problems creators hit with AI-generated audio.

These are deterministic DSP measurements (not ML): silence / missing audio, clipping,
mains hum, broadband hiss, rumble & DC offset, band-limited ("muffled") audio, edge
clicks, stereo phase problems and loudness. They feed the quality report and the Noise
category of the timeline, and they configure the noise-removal chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.schemas import Loudness, MediaInfo, QualityIssue, Segment
from . import io as aio

FRAME_S = 0.1
EPS = 1e-12


@dataclass
class NoiseProfile:
    hum_hz: float | None = None
    hum_strength_db: float = 0.0
    hiss: bool = False
    hiss_level_db: float | None = None
    rumble: bool = False
    dc_offset: float = 0.0
    clipping_regions: list[tuple[float, float]] = field(default_factory=list)
    bandwidth_hz: float | None = None
    peak_db: float = -120.0

    @property
    def has_removable_noise(self) -> bool:
        return self.hiss or self.hum_hz is not None or self.rumble or abs(self.dc_offset) > 0.01


def _db(x: float | np.ndarray) -> np.ndarray | float:
    return 20 * np.log10(np.maximum(np.abs(x), EPS))


def _regions(mask: np.ndarray, hop: float, min_gap: float = 0.2, min_len: float = 0.0) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return out
    start = prev = idx[0]
    for i in idx[1:]:
        if (i - prev) * hop > min_gap:
            out.append((start * hop, (prev + 1) * hop))
            start = i
        prev = i
    out.append((start * hop, (prev + 1) * hop))
    return [(a, b) for a, b in out if b - a >= min_len]


def analyze(master: Path, media: MediaInfo, loud: Loudness | None = None) -> tuple[list[QualityIssue], list[Segment], NoiseProfile]:
    sr, ch, n = aio.info(master)
    duration = n / sr if sr else 0.0
    issues: list[QualityIssue] = []
    segs: list[Segment] = []
    prof = NoiseProfile()

    if not media.has_audio:
        issues.append(QualityIssue(
            kind="no_audio_track", severity="error", title="No audio track",
            detail="The file has no audio stream. Some generators (and some upscalers) drop audio entirely; "
            "regenerate with audio enabled or add audio in your editor.",
        ))
        return issues, segs, prof

    hop = int(FRAME_S * sr)
    nfft_long = 32768
    welch = np.zeros(nfft_long // 2 + 1)
    welch_count = 0
    rms_l: list[float] = []
    clip_l: list[int] = []
    hf_l: list[float] = []
    flat_l: list[float] = []
    sum_x = 0.0
    corr_num = corr_a = corr_b = 0.0
    identical = True
    peak = 0.0
    first_sample = last_sample = 0.0
    freqs_short = np.fft.rfftfreq(4096, 1 / sr)
    hf_band = (freqs_short >= 4000) & (freqs_short <= min(16000, sr / 2 - 100))
    win_s = np.hanning(4096)
    win_l = np.hanning(nfft_long)

    for pos, block in aio.iter_blocks(master, hop * 100):
        mono = block.mean(axis=1)
        if pos == 0 and len(mono):
            first_sample = float(np.max(np.abs(block[: max(1, int(0.002 * sr))])))
        last_sample = float(np.max(np.abs(block[-max(1, int(0.002 * sr)):])))
        peak = max(peak, float(np.max(np.abs(block))) if len(block) else 0.0)
        sum_x += float(block.sum())
        if ch >= 2:
            a, b = block[:, 0].astype(np.float64), block[:, 1].astype(np.float64)
            corr_num += float(np.dot(a, b))
            corr_a += float(np.dot(a, a))
            corr_b += float(np.dot(b, b))
            if identical and not np.allclose(a, b, atol=1e-5):
                identical = False
        for f0 in range(0, len(mono) - nfft_long + 1, nfft_long):
            seg = mono[f0 : f0 + nfft_long] * win_l
            welch += np.abs(np.fft.rfft(seg)) ** 2
            welch_count += 1
        for f0 in range(0, len(mono), hop):
            fr = mono[f0 : f0 + hop]
            if len(fr) < hop // 2:
                continue
            rms_l.append(float(np.sqrt(np.mean(fr**2) + EPS)))
            frb = block[f0 : f0 + hop]
            clip_l.append(int(np.sum(np.abs(frb) >= 0.999)))
            spec_in = np.zeros(4096, dtype=np.float32)
            m = min(4096, len(fr))
            spec_in[:m] = fr[:m]
            mag = np.abs(np.fft.rfft(spec_in * win_s)) ** 2
            hb = mag[hf_band] + EPS
            hf_l.append(float(10 * np.log10(np.mean(hb) / (np.sum(win_s**2)) + EPS)))
            flat_l.append(float(np.exp(np.mean(np.log(hb))) / np.mean(hb)))

    rms = np.array(rms_l)
    rms_db = np.asarray(_db(rms))
    prof.peak_db = float(_db(peak))
    prof.dc_offset = sum_x / max(1, n * ch)

    # --- silence / missing audio ------------------------------------------------
    if peak < 10 ** (-60 / 20):
        issues.append(QualityIssue(
            kind="silent_audio", severity="error", title="Audio track is silent",
            detail="The audio stream exists but contains (near-)digital silence. This is a known failure mode "
            "of some video generators; regenerate the clip or add audio externally.",
            metrics={"peak_dbfs": round(prof.peak_db, 1)},
        ))
        return issues, segs, prof
    silent = rms_db < -60
    for d0, d1 in _regions(silent, FRAME_S, min_gap=0.0, min_len=1.5):
        if d0 > 0.05 and d1 < duration - 0.05:
            issues.append(QualityIssue(
                kind="dead_air", severity="info", title="Dead air",
                detail=f"{d1 - d0:.1f}s of near-silence inside the clip.", start=d0, end=d1,
            ))

    # --- clipping -----------------------------------------------------------------
    clip = np.array(clip_l) >= 3
    regions = _regions(clip, FRAME_S, min_gap=0.3)
    prof.clipping_regions = regions
    if regions:
        total = sum(b - a for a, b in regions)
        issues.append(QualityIssue(
            kind="clipping", severity="warning" if total < 1 else "error", title="Clipping / distortion",
            detail=f"Samples hit full scale in {len(regions)} region(s), {total:.1f}s total. Clipping is baked "
            "into the waveform: it can be marked but not cleanly removed by separation.",
            start=regions[0][0], end=regions[-1][1], metrics={"regions": len(regions), "seconds": round(total, 2)},
        ))
        for r0, r1 in regions:
            segs.append(Segment(
                category="noise", start=float(r0), end=float(min(r1, duration)), confidence=0.95, stream="mix",
                description="Clipping (samples at full scale)", flags=["clipping", "not_removable"],
            ))

    # --- long-term spectrum: hum, rumble, bandwidth ---------------------------------
    if welch_count:
        spec = welch / welch_count
        freqs = np.fft.rfftfreq(nfft_long, 1 / sr)
        spec_db = 10 * np.log10(spec + EPS)
        best: tuple[float, float] | None = None
        for mains in (50.0, 60.0):
            hits: int = 0
            fund: float = 0.0
            strength: list[float] = []
            for k in range(1, 6):
                f = mains * k
                i = int(round(f / (freqs[1])))
                lo, hi = max(1, i - 40), min(len(spec_db) - 1, i + 40)
                local = np.median(spec_db[lo:hi])
                pk = np.max(spec_db[max(1, i - 1) : i + 2])
                # a mains hum line is narrow: the peak must sit on the exact mains frequency
                narrow = pk - max(spec_db[max(1, i - 4)], spec_db[min(len(spec_db) - 1, i + 4)]) > 6
                if pk - local > 12 and narrow:
                    hits += 1
                    strength.append(float(pk - local))
                    if k == 1:
                        fund = float(pk - local)
            if (hits >= 2 or fund >= 12) and strength and (best is None or np.mean(strength) > best[1]):
                best = (mains, float(np.mean(strength)))
        if best:
            prof.hum_hz, prof.hum_strength_db = best
            issues.append(QualityIssue(
                kind="hum", severity="warning", title=f"Mains hum ({best[0]:.0f} Hz)",
                detail=f"Tonal hum at {best[0]:.0f} Hz and harmonics ({best[1]:.0f} dB above the local spectrum). "
                "Removable with the Noise category (notch filters).",
                fix="remove_noise", metrics={"fundamental_hz": best[0], "prominence_db": round(best[1], 1)},
            ))
        total_e = float(np.sum(spec[freqs > 20])) + EPS
        low_e = float(np.sum(spec[(freqs > 5) & (freqs < 35)]))
        if low_e / total_e > 0.08:
            prof.rumble = True
            issues.append(QualityIssue(
                kind="rumble", severity="warning", title="Low-frequency rumble",
                detail="Significant energy below 35 Hz (rumble/handling noise). Removable with the Noise category.",
                fix="remove_noise", metrics={"ratio": round(low_e / total_e, 3)},
            ))
        smooth = np.convolve(spec_db, np.ones(31) / 31, mode="same")
        ref = np.max(smooth[(freqs > 100) & (freqs < 4000)])
        above = np.flatnonzero((smooth > ref - 70) & (freqs > 1000))
        if len(above):
            bw = float(freqs[above[-1]])
            prof.bandwidth_hz = bw
            if bw < 15000 and sr >= 32000:
                issues.append(QualityIssue(
                    kind="band_limited", severity="info", title=f"Band-limited audio (~{bw / 1000:.1f} kHz)",
                    detail="There is almost no content above this frequency, so the audio can sound muffled. "
                    "Generators often synthesize audio at 16–24 kHz and upsample it. Separation cannot restore "
                    "missing highs; consider a bandwidth-extension tool or regenerating.",
                    metrics={"bandwidth_hz": round(bw)},
                ))
    if abs(prof.dc_offset) > 0.01:
        issues.append(QualityIssue(
            kind="dc_offset", severity="warning", title="DC offset",
            detail=f"The waveform is offset by {prof.dc_offset:+.3f}. Removed by the Noise chain's high-pass.",
            fix="remove_noise",
        ))

    # --- hiss: high band floor in the quietest frames --------------------------------
    if len(hf_l) >= 10:
        hf = np.array(hf_l)
        flat = np.array(flat_l)
        active = rms_db > -70
        if np.any(active):
            q = np.percentile(rms_db[active], 15)
            quiet = active & (rms_db <= q)
            floor = float(np.median(hf[quiet])) if np.any(quiet) else -120.0
            flatness = float(np.median(flat[quiet])) if np.any(quiet) else 0.0
            prof.hiss_level_db = floor
            if floor > -78 and flatness > 0.35:
                prof.hiss = True
                issues.append(QualityIssue(
                    kind="hiss", severity="warning", title="Broadband hiss",
                    detail=f"Stationary high-frequency noise floor (~{floor:.0f} dB per bin, flatness {flatness:.2f}) "
                    "audible in quiet passages. Removable with the Noise category (spectral gating).",
                    fix="remove_noise", metrics={"floor_db": round(floor, 1), "flatness": round(flatness, 2)},
                ))

    if prof.hiss or prof.hum_hz or prof.rumble:
        kinds = [k for k, on in (("hiss", prof.hiss), ("hum", prof.hum_hz), ("rumble", prof.rumble)) if on]
        segs.append(Segment(
            category="noise", start=0.0, end=duration, confidence=0.8, stream="noise",
            description="Stationary " + " + ".join(kinds), flags=kinds,
        ))

    # --- edges, stereo ---------------------------------------------------------------------
    if first_sample > 0.05 or last_sample > 0.05:
        issues.append(QualityIssue(
            kind="edge_click", severity="info", title="Abrupt start/end",
            detail="Audio starts or ends at a non-zero level, which clicks when clips are joined. Exports apply "
            "short edge fades (configurable).", fix="edge_fade",
        ))
    if ch >= 2:
        corr = corr_num / (np.sqrt(corr_a * corr_b) + EPS)
        if identical:
            issues.append(QualityIssue(kind="dual_mono", severity="info", title="Dual-mono audio",
                                       detail="Left and right channels are identical."))
        elif corr < -0.3:
            issues.append(QualityIssue(
                kind="phase", severity="warning", title="Stereo phase problem",
                detail=f"L/R correlation is {corr:.2f}; content may cancel when played in mono (phones).",
                metrics={"correlation": round(float(corr), 2)},
            ))

    # --- loudness -------------------------------------------------------------------------
    if loud and loud.integrated_lufs is not None:
        il = loud.integrated_lufs
        if il < -24 or il > -9:
            issues.append(QualityIssue(
                kind="loudness", severity="info", title=f"Loudness {il:.1f} LUFS",
                detail="Outside typical delivery targets (−14 LUFS for YouTube/TikTok/Reels). Clips from "
                "generators vary a lot; normalise on export so stitched clips match.",
                fix="loudness_preset", metrics=loud.model_dump(),
            ))
        if loud.true_peak_dbtp is not None and loud.true_peak_dbtp > -0.5:
            issues.append(QualityIssue(
                kind="true_peak", severity="warning", title="True peak above −0.5 dBTP",
                detail="Likely to distort after platform transcoding. Use a loudness preset on export.",
                fix="loudness_preset", metrics=loud.model_dump(),
            ))
    return issues, segs, prof
