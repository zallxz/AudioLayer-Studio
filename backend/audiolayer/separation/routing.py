"""Build the four category *streams* from separator output + classifier evidence.

    voice  = clean(vocals)        · (1 − w_sing)
    music  = clean(accompaniment) · w_mus  +  clean(vocals) · w_sing
    fx     = clean(accompaniment) · (1 − w_mus)
    noise  = (vocals − clean(vocals)) + (accompaniment − clean(accompaniment))

so ``voice + music + fx + noise == master`` exactly. ``w_sing`` / ``w_mus`` are smooth
time weights from the classifier run on the separated signals.

Inside time regions where the classifier hears sound effects *on top of* music, the
accompaniment is additionally split in the time–frequency domain with a REPET-SIM style
soft mask (Rafii & Pardo 2012): music is self-similar (repeating), one-off effects are not.
This is a classical separation algorithm and its quality is limited — those regions are
flagged as estimates in the results, never presented as clean.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..audio import io as aio
from ..classification.base import FrameProbs

EPS = 1e-6


def weights(fp_voice: FrameProbs | None, fp_acc: FrameProbs) -> dict[str, np.ndarray]:
    m, f = fp_acc.probs["music"], fp_acc.probs["sfx"]
    act = m + f
    w_mus = np.where(act > 0.1, m / (act + EPS), 0.0)
    if fp_voice is not None:
        s, d = fp_voice.probs["singing"], fp_voice.probs["dialogue"]
        n = min(len(s), len(w_mus))
        vact = s[:n] + d[:n]
        w_sing = np.zeros_like(w_mus)
        w_sing[:n] = np.where(vact > 0.1, s[:n] / (vact + EPS), 0.0)
        # require singing to be backed by music being present in the accompaniment
        w_sing[:n] *= np.clip(m[:n] * 2, 0, 1)
    else:
        w_sing = np.zeros_like(w_mus)
    smooth = max(1, int(round(0.6 / fp_acc.hop)))
    return {
        "w_mus": np.clip(uniform_filter1d(w_mus, smooth), 0, 1),
        "w_sing": np.clip(uniform_filter1d(w_sing, smooth), 0, 1),
        "overlap": (m > 0.3) & (f > 0.15),
        "w_event": np.clip(uniform_filter1d((f - 0.08) / 0.15, smooth), 0, 1),
    }


def tf_foreground(x: np.ndarray, sr: int) -> np.ndarray | None:
    """Non-repeating (foreground) part of ``x`` via nearest-neighbour median filtering.

    Returns ``None`` when librosa (part of the ``ml`` extra) is not installed.
    """
    try:
        import librosa
    except ImportError:
        return None

    n_fft, hop = 2048, 512
    mono = x.mean(axis=1)
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop))
    width = max(3, int(round(1.5 * sr / hop)))
    if S.shape[1] <= width * 2:
        return np.zeros_like(x)
    bg = librosa.decompose.nn_filter(S, aggregate=np.median, metric="cosine", width=width)
    bg = np.minimum(S, bg)
    mask_fg = librosa.util.softmask(S - bg, 2.0 * bg, power=2)
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        Z = librosa.stft(x[:, c], n_fft=n_fft, hop_length=hop)
        out[:, c] = librosa.istft(Z * mask_fg, hop_length=hop, length=len(x))
    return out.astype(np.float32)


def interp(track: np.ndarray, hop: float, start: int, n: int, sr: int) -> np.ndarray:
    t = (start + np.arange(n)) / sr
    centers = (np.arange(len(track)) + 0.5) * hop
    if len(track) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.interp(t, centers, track).astype(np.float32)


def build_streams(
    vocals: Path, accomp: Path, out_dir: Path, w: dict[str, np.ndarray], hop: float,
    clean: Callable[[np.ndarray], np.ndarray] | None, *, block_s: float = 30.0,
    on_progress: Callable[[float], None] | None = None, check_cancel: Callable[[], None] | None = None,
) -> dict[str, Path]:
    sr, ch, n = aio.info(vocals)
    paths = {k: out_dir / f"{k}.wav" for k in ("voice", "music", "fx", "noise")}
    writers = {k: aio.open_writer(p, sr, ch) for k, p in paths.items()}
    block = int(block_s * sr)
    ctx = int(1.0 * sr)  # filter context so block edges stay seamless
    try:
        for s0 in range(0, n, block):
            if check_cancel:
                check_cancel()
            s1 = min(n, s0 + block)
            a, b = max(0, s0 - ctx), min(n, s1 + ctx)
            v = aio.read_range(vocals, a, b)
            acc = aio.read_range(accomp, a, b)
            if clean is not None:
                vc, ac = clean(v), clean(acc)
            else:
                vc, ac = v, acc
            ac_full = ac
            sl = slice(s0 - a, s1 - a)
            v, acc, vc, ac = v[sl], acc[sl], vc[sl], ac[sl]
            m = s1 - s0
            ws = interp(w["w_sing"], hop, s0, m, sr)[:, None]
            wm = interp(w["w_mus"], hop, s0, m, sr)[:, None]
            fx = ac * (1 - wm)
            if "w_event" in w:
                we = interp(w["w_event"], hop, s0, m, sr)[:, None]
                if float(np.max(we)) > 0.05:
                    # TF split needs the full block (with context) to find repetitions
                    fg = tf_foreground(ac_full, sr)
                    if fg is not None:
                        fx = we * fg[sl] + (1 - we) * fx
            writers["voice"].write(vc * (1 - ws))
            writers["music"].write(ac - fx + vc * ws)
            writers["fx"].write(fx)
            writers["noise"].write((v - vc) + (acc - ac))
            if on_progress:
                on_progress(s1 / n)
    finally:
        for wr in writers.values():
            wr.close()
    return paths
