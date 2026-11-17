#!/usr/bin/env python3
"""Export the PyTorch models to ONNX for the torch-free "cloud" engine (maintainers only).

    python scripts/export_onnx.py OUT_DIR

Produces:
  panns_cnn14_sed.onnx   PANNs Cnn14_DecisionLevelMax, waveform(32 kHz) -> framewise probabilities
  htdemucs_core.onnx     Demucs htdemucs network for one 343980-sample segment; STFT/iSTFT are done
                         in numpy by audiolayer.separation.demucs_onnx (ONNX has no complex iSTFT)
Weights are stored as float16 and cast to float32 inside the graph (half the download, fp32 math).
Requires the `ml` extra plus `onnx`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper


def fp16_storage(src: Path, dst: Path, min_elems: int = 1024) -> None:
    """Store large float32 initializers as float16 + Cast(to float32) nodes."""
    model = onnx.load(str(src))
    g = model.graph
    new_inits, casts = [], []
    for init in list(g.initializer):
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= min_elems:
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "__fp16")
            new_inits.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT,
                                          name=init.name + "__cast"))
            g.initializer.remove(init)
    g.initializer.extend(new_inits)
    for i, n in enumerate(casts):
        g.node.insert(i, n)
    onnx.save(model, str(dst))


class DemucsCore(torch.nn.Module):
    """htdemucs forward pass with the STFT/iSTFT and complex ops moved outside the graph.

    Inputs:  mix (1, 2, L) and its complex-as-channels spectrogram mag (1, 4, F, T)
    Outputs: x  (1, S, 4, F, T) masked spectrogram (real/imag as channels) and xt (1, S, 2, L)
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.m = model

    def forward(self, mix: torch.Tensor, mag: torch.Tensor):  # type: ignore[no-untyped-def]
        m = self.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_upsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                x = x.reshape(b, -1, f * t)
                x = m.channel_downsampler(x)
                x = x.reshape(b, c, f, t)
                xt = m.channel_downsampler_t(xt)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, mix.shape[-1])
        xt = xt * stdt[:, None] + meant[:, None]
        return x, xt


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    from audiolayer.models.panns_cnn14 import Cnn14DecisionLevelMax

    ck = torch.load(str(Path.home() / ".cache/audiolayer/panns/Cnn14_DecisionLevelMax_mAP=0.385.pth"),
                    map_location="cpu", weights_only=True)
    pm = Cnn14DecisionLevelMax()
    pm.load_state_dict(ck.get("model", ck), strict=False)
    pm.eval()
    tmp = out / "_panns_fp32.onnx"
    torch.onnx.export(pm, (torch.randn(1, 320000) * 0.1,), str(tmp), input_names=["waveform"], output_names=["probs"],
                      dynamic_axes={"waveform": {1: "n"}, "probs": {1: "t"}}, opset_version=17, dynamo=False)
    fp16_storage(tmp, out / "panns_cnn14_sed.onnx")
    tmp.unlink()

    from demucs.pretrained import get_model

    bag = get_model("htdemucs")
    dm = bag.models[0] if hasattr(bag, "models") else bag
    dm.eval()
    L = int(dm.segment * dm.samplerate)
    mix = torch.randn(1, 2, L) * 0.1
    from audiolayer.separation.demucs_onnx import spec_cac

    mag = torch.from_numpy(spec_cac(mix.numpy(), dm.nfft, dm.hop_length))
    tmp = out / "_htdemucs_fp32.onnx"
    torch.onnx.export(DemucsCore(dm), (mix, mag), str(tmp), input_names=["mix", "mag"], output_names=["x", "xt"],
                      opset_version=17, dynamo=False)
    fp16_storage(tmp, out / "htdemucs_core.onnx")
    tmp.unlink()
    print("segment length", L, "sources", dm.sources)


if __name__ == "__main__":
    main()
