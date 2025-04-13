# -*- coding: utf-8 -*-

import os
import yaml
from pathlib import Path

class ConfigManager:
    _instance = None
    _config = {}
    _initialized = False

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def initialize(self, config_path=None, config_dict=None):
        """初始化配置"""
        if self._initialized:
            return
        
        self._config = {
            'DATA_CONFIG': DATA_CONFIG,
            'MODEL_CONFIG': MODEL_CONFIG,
            'TRAIN_CONFIG': TRAIN_CONFIG,
            'AUDIO_CONFIG': AUDIO_CONFIG,
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self._update_nested_dict(self._config, config)

        if config_dict:
            self._update_nested_dict(self._config, config_dict)

        self._initialized = True

    def _update_nested_dict(self, d, u):
        """更新嵌套字典"""
        for k, v in u.items():
            if isinstance(v, dict):
                if k not in d:
                    d[k] = {}
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def get(self, key, default=None):
        """获取配置"""
        return self._config.get(key, default)
    
    def get_nested(self, *keys, default=None):
        """获取嵌套配置"""
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
config = ConfigManager()

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()

# 数据相关配置
DATA_CONFIG = {
    'data_dir': os.path.join(ROOT_DIR, 'data'),
    'cover_dir_name': 'cover', # cover音频目录名称
    'train_ratio': 0.8, # 训练集占完整数据集的比例
    'batch_size': 32,   # 批次大小
    'shuffle_buffer': 1000, # 数据加载缓冲区大小
    'num_workers': 4,   # 数据加载线程数
    'max_samples': None,  # 最大样本数
}

# 模型相关配置
MODEL_CONFIG = {
    'input_channels': 1,
    'input_shape': (None, None, 1),  # 音频输入形状
    'learning_rate': 0.001,
}

# 训练相关配置
TRAIN_CONFIG = {
    'num_epochs': 20,
    'early_stopping_patience': 10,
    'model_checkpoint_dir': os.path.join(ROOT_DIR, 'models'),
    'tensorboard_log_dir': os.path.join(ROOT_DIR, 'logs'),
}

# 音频处理相关配置
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'duration': 2,  # 音频片段长度(秒)
    'frame_length': 256,
    'hop_percentage': 0.5,
    'target_bitrate': '512k',
    'target_profile': 'aac_he_v2'
}

def load_config(config_path):
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    """保存配置到YAML文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
