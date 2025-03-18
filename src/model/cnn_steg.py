# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import svm
from sklearn.preprocessing import StandardScaler

class ResidualUnit(nn.Module):
    """残差单元，包含两层卷积层和身份映射"""
    def __init__(self, channels):
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out - identity  # H(x) = F(x) - x
        out = self.relu(out)
        return out

class DPES(nn.Module):
    """双流金字塔增强模块"""
    def __init__(self, channels):
        super(DPES, self).__init__()
        self.os_branch = self._make_branch(channels)  # 原始音频分支
        self.cs_branch = self._make_branch(channels)  # 隐写音频分支
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fusion = nn.Linear(channels * 2, channels)

    def _make_branch(self, channels):
        layers = [ResidualUnit(channels) for _ in range(5)]
        return nn.Sequential(*layers)

    def forward(self, o_spectrogram, c_spectrogram):
        os_features = self.os_branch(o_spectrogram)
        cs_features = self.cs_branch(c_spectrogram)
        
        # 全局平均池化
        os_features = self.global_avg_pool(os_features).view(os_features.size(0), -1)
        cs_features = self.global_avg_pool(cs_features).view(cs_features.size(0), -1)
        
        # 特征拼接
        combined = torch.cat((os_features, cs_features), dim=1)
        
        # 线性变换
        enhanced_features = self.fusion(combined)
        return enhanced_features

class CNNStegAnalysis(nn.Module):
    def __init__(self, input_channels=3):
        super(CNNStegAnalysis, self).__init__()
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 卷积层 A 32通道，5个残差单元
        self.convA = self._make_group(32, 5)
        
        # 卷积层 B 64通道，5个残差单元
        self.convB = self._make_group(64, 5)
        
        # 卷积层 C 128通道，5个残差单元
        self.convC = self._make_group(128, 5)
        
        # 双流金字塔增强模块
        self.dpes = DPES(128)
        
        # 特征标准化
        self.scaler = StandardScaler()
        
        # SVM 分类器
        self.svm = svm.SVC(probability=True)
    
    def _make_group(self, channels, num_units):
        """创建卷积组，包含多个残差单元和一次平均池化"""
        layers = [ResidualUnit(channels) for _ in range(num_units)]
        layers.append(nn.AvgPool2d(kernel_size=3, stride=2))  # 平均池化层
        return nn.Sequential(*layers)
    
    def forward(self, inputs):
        o_spectrogram, c_spectrogram = inputs
        
        # 初始卷积层
        o_spectrogram = self.initial_conv(o_spectrogram)
        c_spectrogram = self.initial_conv(c_spectrogram)
        
        # 卷积层 A
        o_spectrogram = self.convA(o_spectrogram)
        c_spectrogram = self.convA(c_spectrogram)
        
        # 卷积层 B
        o_spectrogram = self.convB(o_spectrogram)
        c_spectrogram = self.convB(c_spectrogram)
        
        # 卷积层 C
        o_spectrogram = self.convC(o_spectrogram)
        c_spectrogram = self.convC(c_spectrogram)
        
        # 双流金字塔增强模块
        enhanced_features = self.dpes(o_spectrogram, c_spectrogram)
        
        # 展平
        x = enhanced_features.view(enhanced_features.size(0), -1)
        
        # 特征标准化
        x = self.scaler.transform(x.cpu().detach().numpy())
        
        # SVM 分类
        x = self.svm.predict_proba(x)
        
        return torch.tensor(x, dtype=torch.float32).to(inputs[0].device)