import torch
import torchaudio
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
import os
from spectrogram_processing import SPM

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
        for label in ['cover', 'stego']:  # cover: 原始音频, stego: 隐写音频
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
        waveform, sample_rate = torchaudio.load(sample['path'])

        spm = SPM()
        
        # 频谱图转化
        mel_spectrogram = spm.process(waveform, sample_rate)
        
        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)
        
        return mel_spectrogram, sample['label']

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    创建数据加载器
    Args:
        data_dir (str): 数据目录路径
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
    Returns:
        train_loader, val_loader: 训练集和验证集的数据加载器
    """
    # 数据标准化转换
    transform = torch.nn.Sequential(
        torchaudio.transforms.Normalize(mean=-27.51, std=16.59)  # 这些值需要根据实际数据集调整
    )
    
    # 创建数据集
    full_dataset = AudioStegDataset(data_dir, transform=transform)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
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
