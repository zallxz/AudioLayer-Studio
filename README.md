# AudioLayer Studio — model weight mirror

This branch only hosts model weights for AudioLayer's torch-free "cloud" engine (ONNX Runtime +
CTranslate2), so it can run in sandboxes that can reach GitHub but not Hugging Face / Zenodo.
Files are split into ≤45 MB parts; `audiolayer` downloads, joins and verifies them (SHA-256 in
`manifest.json` and in `backend/audiolayer/models/registry.py`). Do not merge this branch.

| File | Model | Original source | License |
|---|---|---|---|
| `panns_cnn14_sed.onnx` | PANNs Cnn14_DecisionLevelMax (Kong et al. 2020), exported to ONNX, fp16-stored weights | https://zenodo.org/record/3987831 · https://github.com/qiuqiangkong/audioset_tagging_cnn | MIT |
| `htdemucs_core.onnx` | Demucs v4 htdemucs network (Défossez / Meta AI), exported without STFT, fp16-stored weights | https://github.com/adefossez/demucs | MIT |
| `whisper-base/*` | OpenAI Whisper base, CTranslate2 conversion by SYSTRAN | https://huggingface.co/Systran/faster-whisper-base · https://github.com/openai/whisper | MIT |

Export script: `scripts/export_onnx.py` on the `main` branch.
