# 音频隐写分析系统

基于深度学习的音频隐写分析系统，使用CNN模型对音频文件进行分析，判断是否包含隐写信息。

## 项目结构

```
.
├── configs/            # 配置文件目录
├── data/              # 数据目录
│   ├── cover/         # 原始音频文件
│   └── stego/         # 隐写音频文件
├── docs/              # 文档目录
├── model/             # 模型定义
│   └── cnn_steg.py    # CNN模型架构
├── outputs/           # 模型输出和保存目录
├── src/               # 源代码
│   └── data_processing.py  # 数据处理模块
├── tests/             # 测试代码
├── tools/             # 工具脚本
├── requirements.txt   # 项目依赖
├── train.py          # 训练脚本
└── README.md         # 项目说明
```

## 环境要求

- Python 3.8+
- PyTorch==2.6.0
- CUDA 12.6
- 其他依赖见 requirements.txt

## 安装

1. 克隆项目：
```bash
git clone https://github.com/lingshijunrenzy/CNNAudioSteganalysis.git
cd CNNAudioSteganalysis
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

1. 在 `data/` 目录下创建以下子目录：
   - `cover/`: 存放原始音频文件
   - `stego/`: 存放隐写音频文件

2. 将音频文件（.wav 或 .mp3 格式）放入相应目录

## 训练模型

使用以下命令开始训练：

```bash
python train.py --data_dir data --batch_size 32 --num_epochs 50 --lr 0.001
```

参数说明：
- `--data_dir`: 数据目录路径
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--lr`: 学习率
- `--num_workers`: 数据加载线程数

## 模型结构

本项目使用CNN模型进行音频隐写分析，主要特点：

1. 输入：音频梅尔频谱图
2. 4个卷积块，每个块包含：
   - 2D卷积层
   - 批归一化
   - ReLU激活
   - 最大池化
3. 自适应平均池化
4. 全连接层用于二分类

## 输出

训练过程中的最佳模型将保存在 `outputs/` 目录下：
- `best_model.pth`: 验证集上性能最好的模型权重

## 注意事项

1. 确保音频文件格式统一（建议使用WAV格式）
2. 数据集应保持平衡（正负样本数量接近）
3. 建议使用GPU进行训练

## 许可证

MIT License
