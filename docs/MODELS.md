# Model research, choices and measured results

## What each task needs

| Task | Chosen | License | Why | Alternatives considered |
|---|---|---|---|---|
| Sound event detection (framewise, multi-label, 527 AudioSet classes) | **PANNs CNN14 DecisionLevelMax** (vendored inference code, weights from Zenodo 3987831) | MIT | Framewise output (0.32 s) in one pass — fast on CPU, overlap-aware because labels are independent sigmoids. | **AST** (MIT, via `transformers`): better clip accuracy but needs 10 s windows → ~1.5 s CPU per window, too slow for a sliding timeline. **BEATs** / **EfficientAT**: good, extra integration work. **YAMNet**: TensorFlow. |
| Source separation | **Demucs v4 `htdemucs`** (`htdemucs_ft` selectable) | MIT (code + weights) | Mature, pip-installable, CPU-capable, strong vocal/accompaniment split. | **Bandit / Banquet (DnR, cinematic dialogue/music/effects)** — the right *task*, but weights/licensing and packaging are not turnkey yet; the `Separator` interface is designed so a DnR adapter drops in. **BS/Mel-RoFormer** via `audio-separator` (MIT code; weights have mixed licenses). **Open-Unmix** (MIT, weaker). |
| Music vs. effects inside the accompaniment | Classifier-weighted routing + **REPET-SIM** soft mask in SFX regions (`librosa.decompose.nn_filter`) | ISC (librosa) | Classical, documented algorithm; works when music repeats and effects don't. | A cinematic (DnR) model would replace this. |
| Speech recognition, timestamps, intelligibility | **faster-whisper** (`base` default) | MIT (code + Whisper weights) | CTranslate2 int8 on CPU is fast; VAD built in; log-prob/no-speech/compression-ratio give a gibberish signal. | whisper.cpp, WhisperX (adds alignment, more deps), NeMo Parakeet (strong English, heavier). |
| Noise reduction | scipy high-pass + IIR notch (hum) + **noisereduce** stationary spectral gating | BSD / MIT | Only applied to measured problems; residual becomes the Noise stem, so nothing is lost. | **DeepFilterNet** (MIT/Apache; great for speech, Rust build, speech-only), RNNoise. |
| Fallback (no PyTorch) | DSP heuristic classifier | Apache-2.0 | App still runs; clearly marked low accuracy. | — |

All weights are downloaded on demand (`scripts/download_models.py`), never committed. Total ≈ 550 MB
(PANNs 312 MB, Demucs 84 MB, Whisper base 145 MB).

## Hardware

| Setup | 20 s clip, full analysis (measured on a 2-vCPU cloud VM, CPU only) |
|---|---|
| separation (htdemucs) | ~16 s |
| classification (PANNs, run on both separated signals) | ~2–3 s |
| stream routing incl. REPET-SIM | ~1.5–9 s (depends on SFX regions) |
| transcription (whisper base int8) | ~3–6 s |
| **total** | **≈ 25–32 s** (≈ 1.3–1.6× real time) |

A CUDA GPU makes Demucs ~10–30× faster. Apple-silicon MPS works for Demucs; PANNs runs on CPU there.
Memory is bounded by `AUDIOLAYER_CHUNK_SECONDS` (default 30 s chunks with 2 s cross-faded overlap).

## Measured quality (not assumed)

`scripts/make_demo_media.py` builds a scene with known ground truth (LibriSpeech speech 3–11 s,
a Kevin MacLeod music bed throughout, an impact, birdsong, a whoosh, hiss + 60 Hz hum), then
`scripts/evaluate.py` computes SI-SDR of each exported stem against its truth. Results from the run
used during development (higher is better; "mix" = doing nothing):

| Stem | SI-SDR of stem | SI-SDR of unprocessed mix | Improvement |
|---|---|---|---|
| Dialogue | 6.2 dB | −0.4 dB | **+6.6 dB** |
| BGM (+Music) | 1.5 dB | −2.6 dB | +4.1 dB |
| SFX | −6.7 dB | −7.3 dB | +0.5 dB — the birdsong was found (13.3–15.0 s, true 13–16 s) but the synthetic impact and whoosh were heard as "music" |
| Noise | −17.0 dB | −25.3 dB | +8.3 dB (hiss captured; the very quiet hum, −48 dBFS, was not detected) |

Sum of the four streams equals the extracted master to within 1.2e-7 (float rounding) — removal is
always "original minus what you removed", never a filtered approximation.

What this means in practice: removing **music under dialogue** and **separating voice** works; removing
**effects that overlap music** is an estimate. These numbers come from one synthetic scene; real
generator output varies, so use the A/B toggle and your ears.

## Torch-free "cloud" engine (ONNX Runtime)

`scripts/export_onnx.py` exports PANNs (whole model, waveform in → probabilities out) and the htdemucs
*network* (the complex STFT/iSTFT are re-implemented in numpy in `separation/demucs_onnx.py`, since ONNX
cannot express them). Weights are stored as float16 and cast to float32 in the graph. Checks run while
building it:

| Check | Result |
|---|---|
| PANNs ONNX (fp32) vs PyTorch, framewise probabilities | max abs diff 3e-7 |
| PANNs ONNX fp16-stored weights vs fp32 | max abs diff 7e-4 (int8 quantisation was tried and rejected: it broke the model) |
| htdemucs ONNX + numpy spectrograms vs PyTorch forward, one 7.8 s segment | 80 dB (vocals), 75–82 dB (drums/other), 50 dB (bass) agreement |
| Full pipeline on the demo scene, dialogue SI-SDR improvement | **+7.6 dB** (PyTorch path +6.6 dB) |
| Same scene, BGM / noise improvement | +4.7 dB / +8.4 dB |
| Simulated sandbox: no PyTorch, no system FFmpeg/ffprobe, Hugging Face blocked | full analyze → clean → video export works; 20 s clip ≈ 35 s |

Weights mirror: branch `model-weights` (PANNs 164 MB, htdemucs 91 MB, Whisper base 145 MB; split into
≤45 MB parts). Licenses: all MIT (see that branch's README).
