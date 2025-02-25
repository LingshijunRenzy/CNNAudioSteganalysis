import os
import yaml
from pathlib import Path

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
