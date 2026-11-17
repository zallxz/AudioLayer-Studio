"""Inference-only copy of PANNs ``Cnn14_DecisionLevelMax`` (sound event detection).

Adapted from https://github.com/qiuqiangkong/audioset_tagging_cnn (MIT License,
Copyright (c) 2020 Qiuqiang Kong). Training-only code (mixup, SpecAugment, dropout) was
removed. The architecture and parameter names are unchanged so the official checkpoint
``Cnn14_DecisionLevelMax_mAP=0.385.pth`` (Zenodo record 3987831) loads directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

SAMPLE_RATE = 32000
WINDOW = 1024
HOP = 320
MELS = 64
FMIN = 50
FMAX = 14000
POOL_RATIO = 32  # time downsampling of the CNN
FRAME_HOP_S = HOP * POOL_RATIO / SAMPLE_RATE  # 0.32 s per output frame


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size: tuple[int, int] = (2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14DecisionLevelMax(nn.Module):
    def __init__(self, classes_num: int = 527):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=WINDOW, hop_length=HOP, win_length=WINDOW, window="hann", center=True,
            pad_mode="reflect", freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=SAMPLE_RATE, n_fft=WINDOW, n_mels=MELS, fmin=FMIN, fmax=FMAX, ref=1.0, amin=1e-10,
            top_db=None, freeze_parameters=True,
        )
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (batch, samples) at 32 kHz -> (batch, frames, classes) probabilities."""
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = torch.mean(x, dim=3)
        x = F.max_pool1d(x, kernel_size=3, stride=1, padding=1) + F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        return torch.sigmoid(self.fc_audioset(x))
