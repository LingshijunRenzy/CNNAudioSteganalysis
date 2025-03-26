# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
from src.data_processing import get_dataloaders
from src.model.cnn_steg import CNNStegAnalysis
from src.config import config as cfg

def print_config(config):
    """打印配置信息"""
    print("\n当前配置信息:")
    print("=" * 50)
    
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    print_dict(config)
    print("=" * 50)

def confirm_config():
    """获取用户确认"""
    while True:
        response = input("\nstart train with this config? (Y/N): ").lower()
        if response in ['y', 'n']:
            return response == 'y'

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    训练模型
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备
    """
    best_val_acc = 0.0
    
    # 收集特征和标签用于训练 SVM
    train_features = []
    train_labels = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            # change train input to fit DPES
            o_spectrograms, c_spectrograms = inputs
            o_spectrograms = o_spectrograms.to(device)
            c_spectrograms = c_spectrograms.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model((o_spectrograms, c_spectrograms))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': train_loss/train_total,
                'acc': 100.*train_correct/train_total
            })
            
            # 收集特征和标签
            train_features.append(outputs.cpu().detach().numpy())
            train_labels.append(labels.cpu().detach().numpy())

            if train_bar.n % 100 == 0:
                torch.cuda.empty_cache()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_bar:
                o_spectrograms, c_spectrograms = inputs
                o_spectrograms = o_spectrograms.to(device)
                c_spectrograms = c_spectrograms.to(device)
                labels = labels.to(device)
                
                outputs = model((o_spectrograms, c_spectrograms))
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': val_loss/val_total,
                    'acc': 100.*val_correct/val_total
                })
        
        val_acc = 100.*val_correct/val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存最佳模型
            torch.save(model.state_dict(), 'outputs/best_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/train_total:.4f}, Train Acc: {100.*train_correct/train_total:.2f}%')
        print(f'Val Loss: {val_loss/val_total:.4f}, Val Acc: {100.*val_correct/val_total:.2f}%')
    
    # 训练 SVM 分类器
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    model.scaler.fit(train_features)
    train_features = model.scaler.transform(train_features)
    model.svm.fit(train_features, train_labels)

def main():
    parser = argparse.ArgumentParser(description='音频隐写分析模型训练')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数')
    parser.add_argument('--config', type=str, help='配置文件路径')
    args = parser.parse_args()
    
    cfg.initialize(config_path=args.config)
    
    # 将命令行参数添加到配置中
    runtime_config = {
        'DATA_CONFIG': {
            'data_dir': args.data_dir,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'max_samples': args.max_samples
        },
        'MODEL_CONFIG': {
            'learning_rate': args.lr
        },
        'TRAIN_CONFIG': {
            'num_epochs': args.num_epochs
        }
    }

    cfg._update_nested_dict(cfg._config, runtime_config)

    config = cfg._config
    
    # 打印并确认配置
    print_config(config)
    if not confirm_config():
        print("用户取消训练")
        return
    
    # 创建输出目录
    os.makedirs(config['TRAIN_CONFIG']['model_checkpoint_dir'], exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(
        config['DATA_CONFIG']['data_dir'],
        config['DATA_CONFIG']['batch_size'],
        config['DATA_CONFIG']['num_workers']
    )
    
    # 创建模型
    model = CNNStegAnalysis(
        input_channels=config['MODEL_CONFIG']['input_channels']
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['MODEL_CONFIG']['learning_rate'])
    
    # 训练模型
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config['TRAIN_CONFIG']['num_epochs'],
        device
    )

if __name__ == '__main__':
    main()


