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
                d[k] = self._update_nested_dict(d.get(k, {}), v)
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
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'batch_size': 32,
    'shuffle_buffer': 1000,
}

# 模型相关配置
MODEL_CONFIG = {
    'input_channels': 1,
    'input_shape': (None, None, 1),  # 音频输入形状
    'num_classes': 2,  # 二分类问题
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
}

# 训练相关配置
TRAIN_CONFIG = {
    'epochs': 100,
    'early_stopping_patience': 10,
    'model_checkpoint_dir': os.path.join(ROOT_DIR, 'models'),
    'tensorboard_log_dir': os.path.join(ROOT_DIR, 'logs'),
}

# 音频处理相关配置
AUDIO_CONFIG = {
    'sample_rate': 44100,
    'duration': 10,  # 音频片段长度(秒)
    'hop_length': 512,
    'n_mels': 128,
    'n_fft': 2048,
}

def load_config(config_path):
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    """保存配置到YAML文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
