#!/usr/bin/env python3
"""Measure separation quality of an analyzed project against ground-truth stems.

    python scripts/evaluate.py WORKSPACE PROJECT_ID TRUTH_DIR

TRUTH_DIR is the output folder of scripts/make_demo_media.py. Prints SI-SDR (dB, higher is
better) for each category stem vs. its ground truth, the SI-SDR of the unprocessed mix
as a baseline, and the reconstruction error of (sum of stems) vs. the extracted master.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est, ref = est - est.mean(), ref - ref.mean()
    if np.sum(ref**2) < 1e-10:
        return float("nan")
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = a * ref
    noise = est - target
    return float(10 * np.log10(np.sum(target**2) / (np.sum(noise**2) + 1e-12)))


def mono(p: Path) -> np.ndarray:
    x, _ = sf.read(str(p), dtype="float32", always_2d=True)
    return np.asarray(x.mean(axis=1))


def main() -> None:
    ws, pid, truth = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    pdir = ws / "projects" / pid
    master = mono(pdir / "audio" / "master.wav")
    streams = {k: mono(pdir / "streams" / f"{k}.wav") for k in ("voice", "music", "fx", "noise")}
    n = min(len(master), *(len(v) for v in streams.values()))
    recon = sum(v[:n] for v in streams.values())
    err = float(np.max(np.abs(recon - master[:n])))
    stems = {c: mono(pdir / "stems" / f"{c}.flac") for c in ("dialogue", "bgm", "sfx", "music", "noise")}
    pairs = {"dialogue": ["dialogue"], "bgm": ["bgm", "music"], "sfx": ["sfx"], "noise": ["noise"]}
    result: dict[str, object] = {"sum_of_streams_max_abs_error": err}
    for truth_name, cats in pairs.items():
        f = truth / f"truth_{truth_name}.wav"
        if not f.exists():
            continue
        ref = mono(f)
        m = min(len(ref), n)
        est = sum(stems[c][:m] for c in cats)
        result[truth_name] = {
            "si_sdr_stem_db": round(si_sdr(est, ref[:m]), 2),
            "si_sdr_unprocessed_mix_db": round(si_sdr(master[:m], ref[:m]), 2),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
