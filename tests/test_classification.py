"""Classification output schema, segmentation and timestamp accuracy."""

from __future__ import annotations

import json

import numpy as np

from audiolayer.classification.base import GROUPS, LABEL_CATEGORY, TRACKS, FrameProbs, audioset_labels
from audiolayer.classification.heuristic import HeuristicClassifier
from audiolayer.classification.segmenter import assign_music_role, hysteresis, regions, segment_track
from audiolayer.core.schemas import CATEGORIES, Segment


def _fp(hop: float, n: int, **wins: list[tuple[float, float]]) -> FrameProbs:
    probs = {t: np.zeros(n) for t in TRACKS}
    for k, ws in wins.items():
        for a, b in ws:
            probs[k][int(a / hop) : int(b / hop)] = 0.8
    return FrameProbs(hop=hop, probs=probs)


def test_audioset_mapping_is_complete_and_disjoint():
    labels = audioset_labels()
    assert len(labels) == 527 and labels[0] == "Speech" and labels[137] == "Music"
    assert labels[285] == "Wind noise (microphone)" and 285 in GROUPS["noise"]
    assert 283 not in GROUPS["noise"]  # natural wind is ambience (SFX), not noise
    assert LABEL_CATEGORY[289] == "sfx"  # rain
    assert LABEL_CATEGORY[27] == "singing"


def test_hysteresis_and_regions():
    p = np.array([0, 0.5, 0.2, 0.2, 0.05, 0, 0.5, 0.5, 0])
    m = hysteresis(p, 0.3, 0.1)
    assert m.tolist() == [False, True, True, True, False, False, True, True, False]
    assert regions(m, 1.0, max_gap=1.0, min_len=0) == [(1, 4), (6, 8)]
    assert regions(m, 1.0, max_gap=3.0, min_len=0) == [(1, 8)]


def test_timestamp_accuracy_within_one_frame():
    hop = 0.05
    fp = _fp(hop, 200, dialogue=[(1.23, 4.56)], sfx=[(6.0, 6.3)])
    d = segment_track(fp, "dialogue", "dialogue", "voice", {"dialogue"})
    assert len(d) == 1
    assert abs(d[0].start - 1.23) <= hop and abs(d[0].end - 4.56) <= hop
    s = segment_track(fp, "sfx", "sfx", "fx", {"sfx"})
    assert len(s) == 1 and abs(s[0].start - 6.0) <= hop


def test_overlapping_sources_are_kept():
    fp = _fp(0.1, 100, dialogue=[(2, 5)], music=[(0, 10)])
    d = segment_track(fp, "dialogue", "dialogue", "voice", {"dialogue"})
    m = segment_track(fp, "music", "bgm", "music", {"music"})
    assert d and m and m[0].start < d[0].start < d[0].end < m[0].end


def test_low_confidence_is_marked_uncertain_not_dropped():
    fp = _fp(0.1, 50)
    fp.probs["sfx"][10:20] = 0.3
    s = segment_track(fp, "sfx", "sfx", "fx", {"sfx"})
    assert len(s) == 1 and s[0].uncertain and s[0].confidence < 0.35


def test_music_role_bgm_vs_foreground():
    fp = _fp(0.1, 100, music=[(0, 10)], dialogue=[(0, 8)])
    seg = segment_track(fp, "music", "bgm", "music", {"music"})[0]
    assign_music_role(seg, fp, fp.probs["dialogue"], None)
    assert seg.category == "bgm"
    fp2 = _fp(0.1, 100, music=[(0, 10)], singing=[(0, 10)])
    seg2 = segment_track(fp2, "music", "bgm", "music", {"music"})[0]
    assign_music_role(seg2, fp2, fp2.probs["dialogue"], np.zeros(100))
    assert seg2.category == "music"


def test_segment_schema_roundtrip():
    s = Segment(category="dialogue", start=1.0, end=2.5, confidence=0.7)
    d = json.loads(s.model_dump_json())
    for key in ("id", "category", "start", "end", "duration", "confidence", "uncertain", "description", "transcript", "flags"):
        assert key in d
    assert d["duration"] == 1.5 and d["category"] in CATEGORIES


def test_heuristic_classifier_runs_without_torch(tmp_path):
    from conftest import tone, write_stereo

    f = write_stereo(tmp_path / "h.wav", np.concatenate([np.zeros(48000, np.float32), tone(220, 2.0, 0.3)]))
    fp = HeuristicClassifier().frame_probs(f)
    assert set(fp.probs) == set(TRACKS)
    assert all(np.all((v >= 0) & (v <= 0.6 + 1e-6)) for v in fp.probs.values())
    assert abs(fp.n_frames * fp.hop - 3.0) < 0.2
