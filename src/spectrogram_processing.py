# -*- coding: utf-8 -*-

import torchaudio
import torch

class SPM:
    def process(self, waveform, frame_length=512, hop_percentage=0.5):
        """
        提取频谱图
        Args:
            waveform (Tensor): 波形数据
            frame_length (int): 帧长度，默认为512点
            hop_percentage (float): 帧移重叠比例，默认为0.5（50%重叠）
        Returns:
            spectrogram (Tensor): 频谱图
        """
        
        # 帧移计算
        hop_length = int(frame_length * (1 - hop_percentage))  # 50%重叠
        
        # 使用汉明窗
        # 计算复数频谱
        
        complex_spec = torch.stft(waveform, n_fft=frame_length, hop_length=hop_length,
                         win_length=frame_length, window=torch.hamming_window(frame_length),
                         return_complex=True)
        
        # 计算对数幅度谱: 20*log10(abs(fft))
        log_magnitude_spec = 20 * torch.log10(torch.abs(complex_spec) + 1e-10)  # 添加小值避免log(0)
        
        # 只保留频谱的前半部分
        spectrogram = log_magnitude_spec[..., :(frame_length//2+1), :]
        
        # Handle multi-channel audio - only keep the first channel
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            # If we have multiple channels, only keep the first one
            spectrogram = spectrogram[0:1]
        
        # Ensure output is in (channels, freq_bins, time_steps) format
        if spectrogram.dim() == 2:
            # If we have a single channel in shape (freq_bins, time_steps)
            spectrogram = spectrogram.unsqueeze(0)
        elif spectrogram.dim() == 3 and spectrogram.shape[0] > 1 and spectrogram.shape[0] != min(spectrogram.shape):
            # If channels dimension is not in first position, rearrange
            spectrogram = spectrogram.permute(2, 0, 1)
        
        return spectrogram
