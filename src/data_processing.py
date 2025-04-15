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
from src.file import FileHandler
import ffmpeg

class AudioNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std

class AudioStegDataset(Dataset):
    def __init__(self, data_dir, transform=None, balance_label=True):
        """
        音频隐写分析数据集
        Args:
            data_dir (str): 数据目录路径
            transform (callable, optional): 数据转换函数
        """
        self.data_dir = data_dir
        self.transform = transform
        self.balance_label = balance_label
        self.cover_samples = []
        self.stego_samples = []
        self.samples = []
        self.AUDIO_CONFIG = cfg.get(key='AUDIO_CONFIG')
        self.DATA_CONFIG = cfg.get(key='DATA_CONFIG')
        self.STORAGE_CONFIG = cfg.get(key='STORAGE_CONFIG')

        storage_type = self.STORAGE_CONFIG.get('type', 'local')
        if storage_type == 'oss':
            # 使用OSS存储
            self._load_from_oss(data_dir)
        elif storage_type == 'local':
            # 使用本地存储
            self._load_from_local(data_dir)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
        
        if self.balance_label:
            self._balance_samples()
        else:
            self.samples = self.cover_samples + self.stego_samples
        
        print(f'found {len(self.samples)} samples in {self.data_dir}')

    def _load_from_oss(self, data_dir):
        """从OSS加载数据"""
        try:
            import oss2
        except ImportError:
            raise ImportError("请安装oss2库: pip install oss2")
        
        oss_config = self.STORAGE_CONFIG.get('oss', {})
        auth = oss2.Auth(oss_config.get('access_key_id'), oss_config.get('access_key_secret'))
        bucket = oss2.Bucket(auth, oss_config.get('endpoint'), oss_config.get('bucket_name'))
        
        # 确保OSS路径前缀格式正确
        prefix = self.data_dir.lstrip('/')
        if not prefix.endswith('/'):
            prefix += '/'
        
        # 从OSS列出所有对象
        print(f"从OSS加载数据，前缀: {prefix}")
        
        # 首先收集所有子目录
        subdirs = set()
        for obj in oss2.ObjectIterator(bucket, prefix=prefix, delimiter='/'):
            if obj.key.endswith('/') and obj.key != prefix:
                # 这是一个子目录
                subdir_name = obj.key.rstrip('/').split('/')[-1]
                subdirs.add(subdir_name)
        
        if not subdirs:
            raise ValueError(f"在OSS中未找到任何子目录，前缀: {prefix}")
        
        print(f"在OSS中找到 {len(subdirs)} 个子目录")
        
        # 然后遍历每个子目录收集音频文件
        for subdir in subdirs:
            subdir_prefix = f"{prefix}{subdir}/"
            is_cover = (subdir.lower() == self.DATA_CONFIG['cover_dir_name'].lower())
            label = 0 if is_cover else 1
            
            for obj in oss2.ObjectIterator(bucket, prefix=subdir_prefix):
                if not obj.key.endswith('/'):  # 排除目录对象
                    file_ext = os.path.splitext(obj.key)[1].lower()
                    if file_ext in ('.wav', '.mp3', '.flac', '.ogg', '.aac'):
                        # 构建与本地路径格式一致的路径，以便后续处理
                        # 注意：在OSS模式下，这个路径实际上是一个标识符，而不是实际的文件路径
                        # FileHandler会根据存储类型进行相应处理
                        oss_path = obj.key
                        local_path_format = os.path.join(self.data_dir, subdir, os.path.basename(obj.key))
                        
                        sample = {
                            'path': oss_path,
                            'label': label,
                            'stego_method': subdir if not is_cover else 'cover'
                        }

                        if is_cover:
                            self.cover_samples.append(sample)
                        else:
                            self.stego_samples.append(sample)
        print(f"从OSS加载完成，共 {len(self.cover_samples)} 个封面样本和 {len(self.stego_samples)} 个隐写样本")

    
    def _load_from_local(self, data_dir):
        """从本地文件系统加载数据"""
        # 获取所有子目录
        subdirs = [d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))]
        
        if not subdirs:
            raise ValueError(f"could not find any subdirectories in {self.data_dir}")

        print(f"Found {len(subdirs)} subdirectories in {self.data_dir}")
        
        # 遍历数据目录
        for subdir in subdirs:
            subdir_path = os.path.join(self.data_dir, subdir)
            is_cover = (subdir.lower() == self.DATA_CONFIG['cover_dir_name'].lower())
            label = 0 if is_cover else 1

            for audio_file in os.listdir(subdir_path):
                if audio_file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.aac')):
                    sample = {
                        'path': os.path.join(subdir_path, audio_file),
                        'label': label,
                        'stego_method': subdir if not is_cover else 'cover'
                    }

                    if is_cover:
                        self.cover_samples.append(sample)
                    else:
                        self.stego_samples.append(sample)
        print(f"Loaded {len(self.cover_samples)} cover samples and {len(self.stego_samples)} stego samples from local storage")

    def _balance_samples(self):
        """平衡样本数量，使stego样本数量为cover样本数量的3倍"""
        cover_count = len(self.cover_samples)
        stego_count = len(self.stego_samples)

        print(f"原始数据分布 - Cover: {cover_count}, Stego: {stego_count}")

        if cover_count == 0 or stego_count == 0:
            raise ValueError("至少一种标签的样本数量为0，无法平衡样本")

        # 计算目标比例下的最优样本数
        ratio = 3  # stego:cover = 3:1

        # 计算能够满足3:1比例的最大可能样本数
        max_cover_possible = min(cover_count, stego_count // ratio)
        target_cover_count = max_cover_possible
        target_stego_count = target_cover_count * ratio

        print(f"目标分布 - Cover: {target_cover_count}, Stego: {target_stego_count} (比例 1:{ratio})")

        # 如果原始数据已经符合目标比例，直接返回
        if cover_count == target_cover_count and stego_count == target_stego_count:
            self.samples = self.cover_samples + self.stego_samples
            return

        # 选择cover样本
        if cover_count > target_cover_count:
            # 从cover样本中随机选择目标数量
            indices = torch.randperm(cover_count)[:target_cover_count].tolist()
            final_cover_samples = [self.cover_samples[i] for i in indices]
            print(f"从{cover_count}个cover样本中随机选择{target_cover_count}个")
        else:
            # cover样本数量已经是最优的，全部使用
            final_cover_samples = self.cover_samples

        # 按照隐写方法对stego样本进行分组
        stego_by_method = {}
        for sample in self.stego_samples:
            method = sample['stego_method']
            if method not in stego_by_method:
                stego_by_method[method] = []
            stego_by_method[method].append(sample)

        # 计算每种方法应选择的样本数量
        method_count = len(stego_by_method)

        # 如果stego样本需要减少
        if stego_count > target_stego_count:
            final_stego_samples = []
            
            # 基础分配 - 从每种方法中均匀选择样本
            base_count = target_stego_count // method_count
            remainder = target_stego_count % method_count
            
            # 分配样本数量，确保总数等于target_stego_count
            samples_per_method = {}
            methods = list(stego_by_method.keys())
            for i, method in enumerate(methods):
                if i < remainder:
                    samples_per_method[method] = base_count + 1
                else:
                    samples_per_method[method] = base_count
            
            # 从每种方法中随机选择样本
            for method, count in samples_per_method.items():
                available = len(stego_by_method[method])
                if count > available:
                    print(f"警告: 方法 {method} 只有 {available} 个样本，但需要 {count} 个")
                    samples_per_method[method] = available
                    # 重新分配剩余配额
                    remaining = count - available
                    for other_method in methods:
                        if other_method != method and len(stego_by_method[other_method]) > samples_per_method[other_method]:
                            samples_per_method[other_method] += 1
                            remaining -= 1
                            if remaining == 0:
                                break
                
                # 从当前方法中随机选择样本
                method_indices = torch.randperm(len(stego_by_method[method]))[:samples_per_method[method]].tolist()
                selected = [stego_by_method[method][i] for i in method_indices]
                final_stego_samples.extend(selected)
                print(f"从方法 {method} 的 {len(stego_by_method[method])} 个样本中选择 {len(selected)} 个")
        else:
            # 如果stego样本数量不足以达到3倍cover样本，进一步减少cover样本
            adjusted_target_cover = stego_count // ratio
            if adjusted_target_cover < target_cover_count:
                indices = torch.randperm(len(final_cover_samples))[:adjusted_target_cover].tolist()
                final_cover_samples = [final_cover_samples[i] for i in indices]
                target_stego_count = stego_count
                print(f"由于stego样本不足，进一步减少cover样本至{adjusted_target_cover}个")
            
            # 使用所有stego样本
            final_stego_samples = self.stego_samples

        # 合并最终选择的样本
        self.samples = final_cover_samples + final_stego_samples

        # 更新统计信息
        self.cover_samples = final_cover_samples
        self.stego_samples = final_stego_samples

        # 验证最终比例
        actual_ratio = len(final_stego_samples) / len(final_cover_samples) if len(final_cover_samples) > 0 else float('inf')
        print(f"平衡后 - Cover: {len(final_cover_samples)}, Stego: {len(final_stego_samples)}, 实际比例 1:{actual_ratio:.2f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        file_path = FileHandler.get_file(sample['path'], return_type='path')
        
        # 加载音频
        # original, original_sr = torchaudio.load(sample['path'])
        original, original_sr = torchaudio.load(file_path)
        original = self._resample_if_needed(original, original_sr)

        # 清理音频文件
        if file_path != sample['path']:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
                pass

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
        
        return (o_spectrogram, c_spectrogram), torch.tensor(sample['label'], dtype=torch.long), sample['stego_method']
    
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
            actual_path = FileHandler.get_file(input_path, return_type='path')

            # 提取码率数字部分，用于调整滤波强度
            bitrate_value = int(''.join(filter(str.isdigit, target_bitrate)))
            
            # 根据码率调整低通滤波器频率，模拟不同码率的压缩效果
            # 低码率有更强的低通滤波效果
            # max_freq = min(20000, 10000 + (bitrate_value // 32) * 500)
            nyquist_freq = self.AUDIO_CONFIG['sample_rate'] // 2
            base_freq = min(5000, nyquist_freq // 2)
            additional_freq = min((bitrate_value // 64) * 500, nyquist_freq // 2)
            max_freq = min(nyquist_freq - 100, base_freq + additional_freq)
            
            # 单步处理：直接从输入音频到WAV，应用滤波器模拟压缩/解压缩效果
            wav_data, _ = (
                ffmpeg
                .input(actual_path)
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

            # 清理临时文件
            if actual_path != input_path:
                try:
                    os.remove(actual_path)
                except Exception as e:
                    print(f"Error deleting file {actual_path}: {e}")
                    pass

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

    # 获取存储类型
    storage_config = cfg.get(key='STORAGE_CONFIG', default={'type': 'local'})
    storage_type = storage_config.get('type', 'local')

    # 检查数据目录是否存在
    if storage_type=='local' and not os.path.exists(data_dir):
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
