# -*- coding: utf-8 -*-

import torchaudio
import torch

class SPM:
    def process(self, waveform, sample_rate):
        """
        提取频谱图
        Args:
            waveform (Tensor): 波形数据
            sample_rate (int): 采样率
        Returns:
            spectrogram (Tensor): 频谱图
        """
        
        # 分帧
        frame_length = int(sample_rate * 0.05) # 50ms
        hop_length = int(sample_rate * 0.025) # 25ms

        frames = waveform.unfold(1, frame_length, hop_length)

        # 加窗
        window = torch.hann_window(frame_length)
        frames_windowed = frames * window

        # 傅里叶变换
        spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=frame_length, hop_length=hop_length, power=2)
        spectrogram = spectrogram_transform(frames_windowed)

        # check dim
        if spectrogram.dim() > 3:
            spectrogram = spectrogram.mean(dim=0)

        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
        elif spectrogram.dim() == 3 and spectrogram.shape[0] != 1:
            spectrogram = spectrogram.permute(2, 0, 1)

        return spectrogram
