# CNN音频隐写分析系统

基于CNN和双流金字塔策略的AAC音频隐写分析系统。

## 技术栈

- Python 3.12
- TensorFlow 2.18.0
- 其他依赖库详见 requirements.txt

## 项目结构

```
.
├── src/                    # 源代码目录
│   ├── data_processing.py  # 数据处理模块
│   ├── models.py          # 模型定义
│   ├── train.py           # 训练脚本
│   ├── evaluate.py        # 评估脚本
│   ├── utils.py           # 工具函数
│   └── config.py          # 配置管理
├── data/                   # 数据集目录
├── models/                 # 保存训练模型
├── configs/                # 配置文件
├── tests/                  # 测试代码
├── tools/                  # 工具脚本
├── docs/                   # 文档
└── requirements.txt        # 项目依赖
```

## 使用说明

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 准备数据：
- 将AAC音频数据集放入 `data/` 目录
- 运行数据预处理脚本

3. 训练模型：
```bash
python src/train.py
```

4. 评估模型：
```bash
python src/evaluate.py
```

## 开发计划

- [ ] 实现基础CNN模型架构
- [ ] 实现双流金字塔策略
- [ ] 完成数据预处理模块
- [ ] 实现训练和评估流程
- [ ] 支持多种隐写方法检测
- [ ] 优化模型性能
- [ ] 完善文档和测试
