# -*- coding: utf-8 -*-

import torch
import torchaudio
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
import os
import io
from src.spectrogram_processing import SPM
import ffmpeg

class AudioNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std

class AudioStegDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        音频隐写分析数据集
        Args:
            data_dir (str): 数据目录路径
            transform (callable, optional): 数据转换函数
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # 遍历数据目录
        for label in ['cover', 'lsbee_0.1', 'lsbee_0.2', 'lsbee_0.3', 'lsbee_0.5', 'lsbee_1.0',
                      'min_0.1', 'min_0.2', 'min_0.3', 'min_0.5', 'min_1.0', 
                      'sign_0.1', 'sign_0.2', 'sign_0.3', 'sign_0.5', 'sign_1.0']:  # cover: 原始音频, stego: 隐写音频
            label_dir = os.path.join(data_dir, label)
            if os.path.exists(label_dir):
                for audio_file in os.listdir(label_dir):
                    if audio_file.endswith(('.wav', '.mp3')):
                        self.samples.append({
                            'path': os.path.join(label_dir, audio_file),
                            'label': 0 if label == 'cover' else 1
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载音频
        original, original_sr = torchaudio.load(sample['path'])
        original = self._resample_if_needed(original, original_sr)

        calibration = self.generate_calibration_stream(
            sample['path'],
            target_bitrate='1024k',
            target_profile='aac_he_v2'
        )

        spm = SPM()
        
        # 频谱图转化
        o_spectrogram = spm.process(original)
        c_spectrogram = spm.process(calibration)
        
        return o_spectrogram, c_spectrogram, sample['label']
    
    def _resample_if_needed(self, waveform, original_rate):
        """统一采样率至目标值"""
        if original_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_rate,
                new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
        return waveform
    

    def generate_calibration_stream(self, input_path, target_bitrate='128k', target_profile='aac_lc'):
        """
        使用FFmpeg生成校准流音频
        
        处理流程:
        原始音频 → 解码PCM → AAC重压缩 → 解码为校准流PCM → 返回Tensor
        
        Args:
            input_path: 输入音频路径
            target_bitrate: 目标码率 (如128k)
            target_profile: AAC Profile (如aac_lc)
            
        Returns:
            torch.Tensor: 校准流波形 [channels, samples]
        """
        try:
            # Step1: 解码原始音频并重压缩为AAC
            aac_data, _ = (
                ffmpeg
                .input(input_path)
                .output(
                    'pipe:', 
                    format='adts',
                    acodec='aac',
                    ar=str(self.target_sample_rate),
                    audio_bitrate=target_bitrate,
                    profile=target_profile,
                    cutoff=18000,  # 限制高频带宽避免自动调整
                    ac=1,          # 强制单声道简化处理
                    af="pan=mono|c0=c0"
                )
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            # Step2: 将AAC数据解码回PCM
            wav_data, _ = (
                ffmpeg
                .input('pipe:', format='adts')
                .output(
                    'pipe:',
                    format='wav',
                    acodec='pcm_s16le',
                    ar=str(self.target_sample_rate)
                )
                .run(input=aac_data, capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            # 加载WAV字节流为Tensor
            buffer = io.BytesIO(wav_data)
            waveform, sample_rate = torchaudio.load(buffer)
            return waveform
            
        except ffmpeg.Error as e:
            print(f"[FFmpeg Error] {e.stderr.decode('utf8')}")
            raise RuntimeError(f"校准流生成失败: {input_path}")

def get_dataloaders(data_dir, batch_size, num_workers):
    """
    获取训练和验证数据加载器
    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录 {data_dir} 不存在，请检查路径是否正确。")
    
    # 加载数据集
    train_dataset = ...  # 确保正确加载训练数据集
    val_dataset = ...    # 确保正确加载验证数据集

    # 检查数据集是否为空
    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空，请检查数据目录或数据加载逻辑。")
    if len(val_dataset) == 0:
        raise ValueError("验证数据集为空，请检查数据目录或数据加载逻辑。")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
