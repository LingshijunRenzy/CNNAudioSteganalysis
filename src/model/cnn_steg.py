# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualUnit(nn.Module):
    #残差单元，包含两层卷积层和身份映射
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
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # 二分类：有隐写和无隐写
        )
    
    def _make_group(self, channels, num_units):
        """创建卷积组，包含多个残差单元和一次平均池化"""
        layers = [ResidualUnit(channels) for _ in range(num_units)]
        layers.append(nn.AvgPool2d(kernel_size=3, stride=2))  # 平均池化层
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始卷积层
        x = self.initial_conv(x)
        
        # 卷积层 A
        x = self.convA(x)
        
        # 卷积层 B
        x = self.convB(x)
        
        # 卷积层 C
        x = self.convC(x)
        
        # 全局平均池化
        x = self.global_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x