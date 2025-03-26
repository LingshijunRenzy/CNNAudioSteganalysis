# -*- coding: utf-8 -*-

import torch
import torchaudio
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
import os
import io
from src.spectrogram_processing import SPM
from src.config import config as cfg
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
        self.AUDIO_CONFIG = cfg.get(key='AUDIO_CONFIG')
        self.DATA_CONFIG = cfg.get(key='DATA_CONFIG')

        # 获取所有子目录
        subdirs = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
        
        if not subdirs:
            raise ValueError(f"could not find any subdirectories in {data_dir}")

        print(f"Found {len(subdirs)} subdirectories in {data_dir}")
        
        # 遍历数据目录
        for subdir in subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            is_cover = (subdir.lower() == self.DATA_CONFIG['cover_dir_name'].lower())
            label = 0 if is_cover else 1

            for audio_file in os.listdir(subdir_path):
                if audio_file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.aac')):
                    self.samples.append({
                        'path': os.path.join(subdir_path, audio_file),
                        'label': label,
                        'stego_method': subdir if not is_cover else None
                    })

        print(f"Found {len(self.samples)} audio files in {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载音频
        original, original_sr = torchaudio.load(sample['path'])
        original = self._resample_if_needed(original, original_sr)

        calibration = self.generate_calibration_stream(
            sample['path'],
            target_bitrate=self.AUDIO_CONFIG['target_bitrate'],
            target_profile=self.AUDIO_CONFIG['target_profile'],
        )

        spm = SPM()
        
        # 频谱图转化
        o_spectrogram = spm.process(original,
                                    frame_length=self.AUDIO_CONFIG['frame_length'],
                                    hop_percentage=self.AUDIO_CONFIG['hop_percentage'])
        c_spectrogram = spm.process(calibration,
                                    frame_length=self.AUDIO_CONFIG['frame_length'],
                                    hop_percentage=self.AUDIO_CONFIG['hop_percentage'])
        
        return (o_spectrogram, c_spectrogram), torch.tensor(sample['label'], dtype=torch.long)
    
    def _resample_if_needed(self, waveform, original_rate):
        """统一采样率至目标值"""
        if original_rate != self.AUDIO_CONFIG['sample_rate']:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_rate,
                new_freq=self.AUDIO_CONFIG['sample_rate']
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
            # # Step1: 解码原始音频并重压缩为AAC
            # aac_data, _ = (
            #     ffmpeg
            #     .input(input_path)
            #     .output(
            #         'pipe:', 
            #         format='adts',
            #         acodec='aac',
            #         ar=str(self.AUDIO_CONFIG['sample_rate']),
            #         audio_bitrate=target_bitrate,
            #         profile=target_profile,
            #         cutoff=18000,  # 限制高频带宽避免自动调整
            #         ac=1,          # 强制单声道简化处理
            #         af="pan=mono|c0=c0"
            #     )
            #     .run(capture_stdout=True, capture_stderr=True, quiet=True)
            # )

            # print("第一步编码成功，开始第二步解码...")
            
            # # Step2: 将AAC数据解码回PCM
            # wav_data, _ = (
            #     ffmpeg
            #     .input('pipe:', format='adts')
            #     .output(
            #         'pipe:',
            #         format='wav',
            #         acodec='pcm_s16le',
            #         ar=str(self.AUDIO_CONFIG['sample_rate']),
            #         ac=1,
            #     )
            #     .run(input=aac_data, capture_stdout=True, capture_stderr=True, quiet=True)
            # )

            # 提取码率数字部分，用于调整滤波强度
            bitrate_value = int(''.join(filter(str.isdigit, target_bitrate)))
            
            # 根据码率调整低通滤波器频率，模拟不同码率的压缩效果
            # 低码率有更强的低通滤波效果
            max_freq = min(20000, 10000 + (bitrate_value // 32) * 500)
            
            # 单步处理：直接从输入音频到WAV，应用滤波器模拟压缩/解压缩效果
            wav_data, _ = (
                ffmpeg
                .input(input_path)
                .output(
                    'pipe:',
                    format='wav',  # 使用wav格式而不是adts
                    acodec='pcm_s16le',
                    ar=str(self.AUDIO_CONFIG['sample_rate']),
                    ac=1,  # 强制单声道
                    # 应用滤波器链模拟压缩/解压缩的质量损失
                    af=f"aresample={self.AUDIO_CONFIG['sample_rate']},"\
                    f"highpass=f=20,lowpass=f={max_freq},"\
                    f"acompressor=threshold=0.05:ratio=4:attack=5:release=50"  # 添加压缩器模拟动态范围压缩
                )
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            # 加载WAV字节流为Tensor
            buffer = io.BytesIO(wav_data)
            waveform, sample_rate = torchaudio.load(buffer)
            return waveform
            
        except ffmpeg.Error as e:
            print(f"[FFmpeg Error] 命令执行失败:")
            print(f"标准错误输出: {e.stderr.decode('utf8')}")
            
            # 打印详细的命令信息
            if hasattr(e, 'cmd'):
                print(f"执行的命令: {' '.join(e.cmd)}")
            
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
    
    dataset = AudioStegDataset(data_dir=data_dir)

    DATA_CONFIG = cfg.get(key='DATA_CONFIG')

    max_samples = DATA_CONFIG['max_samples']
    if max_samples and max_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:max_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # 划分数据集
    dataset_size = len(dataset)
    train_size = int(dataset_size * DATA_CONFIG['train_ratio'])
    val_size = dataset_size - train_size
    
    # 加载数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

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
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
