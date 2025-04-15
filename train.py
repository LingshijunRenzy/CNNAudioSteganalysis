# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
import os
from src.data_processing import get_dataloaders
from src.model.cnn_steg import CNNStegAnalysis
from src.config import config as cfg
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast

from src.utils import StegoMethodTracker
from datetime import datetime
import time
import csv

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
    训练模型，并单独跟踪每种隐写方法的性能
    """

    print("开始训练循环...")
    best_val_acc = 0.0
    
    # 收集特征和标签用于训练 SVM
    train_features = []
    train_labels = []

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs = []

    # 创建梯度缩放器用于混合精度训练（使用新API）
    scaler = GradScaler("cuda")

    # 创建隐写方法跟踪器
    train_tracker = StegoMethodTracker()
    val_tracker = StegoMethodTracker() # 添加L2正则化

    # 创建 CSV 文件以保存训练日志
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    csv_path = f'training_log_{timestamp}.csv'

    # 获取所有隐写方法，用于创建CSV表头
    # 训练一个批次以获取可能的隐写方法
    sample_batch = next(iter(train_loader))
    _, _, stego_methods = sample_batch
    all_methods = set()
    for methods in stego_methods:
        if methods:  # 非空字符串
            all_methods.add(methods)

    # 创建CSV文件并写入表头
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'Epoch', 'Timestamp', 'Learning_Rate',
            'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc',
            'Val_Cover_Pred', 'Val_Stego_Pred',
            'Train_Acc_cover', 'Val_Acc_cover'  # 添加cover字段
        ]
    
        # 添加各隐写方法的训练和验证准确率字段
        for method in sorted(all_methods):
            if method and method != 'cover':  # 跳过'cover'，避免重复
                method_lower = method.lower()
                fieldnames.extend([f'Train_Acc_{method_lower}', f'Val_Acc_{method_lower}'])
                
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"training log will be saved to: {csv_path}")

    # 添加学习率调度器
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=15,  # 首次重启周期
    #     T_mult=2,  # 每次重启后周期增加的倍数
    #     eta_min=1e-5,  # 最小学习率
    #     eta_max=5e-5
    # )

    # 替换为OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,        # 30%时间内达到最大学习率
        div_factor=25.0,      # 初始学习率为max_lr/25
        final_div_factor=10000  # 最终学习率为初始学习率/10000
    )
    
    # early stopping
    early_stopping_patience = cfg.get(key='TRAIN_CONFIG')['early_stopping_patience']
    no_improve_epochs = 0
    bast_val_acc = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time() # 计时

        train_tracker.reset_cumulative_stats()
        val_tracker.reset_cumulative_stats()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_bar:
            # 处理包含隐写方法信息的批次数据
            inputs, labels, stego_methods = batch
            o_spectrograms, c_spectrograms = inputs
            o_spectrograms = o_spectrograms.to(device)
            c_spectrograms = c_spectrograms.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            with autocast("cuda"):
                outputs = model((o_spectrograms, c_spectrograms))
                loss = criterion(outputs, labels)

            # 使用scaler进行反向传播和优化器步骤
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # 使梯度裁剪正常工作
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 更新各隐写方法的统计信息
            train_tracker.update(
                predicted.cpu().numpy(), 
                labels.cpu().numpy(), 
                stego_methods
            )
            
            train_bar.set_postfix({
                'loss': train_loss/train_total,
                'acc': 100.*train_correct/train_total
            })
            
            # 收集特征和标签
            if epoch == num_epochs - 1:  # 仅在最后一个epoch收集特征
                train_features.append(outputs.cpu().detach().numpy())
                train_labels.append(labels.cpu().detach().numpy())

            if train_bar.n % 10 == 0:
                torch.cuda.empty_cache()
        
        # 计算本轮epoch的各隐写方法准确率
        train_tracker.compute_epoch_accuracy()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = torch.zeros(2).to(device) 
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_bar:
                # 处理包含隐写方法信息的批次数据
                inputs, labels, stego_methods = batch
                o_spectrograms, c_spectrograms = inputs
                o_spectrograms = o_spectrograms.to(device)
                c_spectrograms = c_spectrograms.to(device)
                labels = labels.to(device)
                
                with autocast("cuda"):
                    outputs = model((o_spectrograms, c_spectrograms))
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # 同时更新预测分布统计
                for p in predicted:
                    val_preds[p] += 1
                
                # 更新各隐写方法的统计信息
                val_tracker.update(
                    predicted.cpu().numpy(), 
                    labels.cpu().numpy(), 
                    stego_methods
                )
                
                val_bar.set_postfix({
                    'loss': val_loss/val_total,
                    'acc': 100.*val_correct/val_total
                })

        # 计算本轮epoch的各隐写方法验证准确率
        val_tracker.compute_epoch_accuracy()

        epochs.append(epoch + 1)
        train_losses.append(train_loss/train_total)
        train_accs.append(100.*train_correct/train_total)
        val_losses.append(val_loss/val_total)
        val_accs.append(100.*val_correct/val_total)
        
        val_acc = 100.*val_correct/val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存最佳模型
            torch.save(model.state_dict(), 'outputs/best_model.pth')

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 创建当前epoch的记录
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'Epoch': epoch + 1,
            'Timestamp': current_time,
            'Learning_Rate': current_lr,
            'Train_Loss': train_loss/train_total,
            'Train_Acc': 100.*train_correct/train_total,
            'Val_Loss': val_loss/val_total,
            'Val_Acc': 100.*val_correct/val_total,
            'Val_Cover_Pred': val_preds[0].item(),
            'Val_Stego_Pred': val_preds[1].item()
        }

        # 添加cover的准确率（调用时传None）
        train_cover_acc = train_tracker.get_cumulative_accuracy(None)
        val_cover_acc = val_tracker.get_cumulative_accuracy(None)

        field_name = 'Train_Acc_cover'
        if train_cover_acc is not None:
            log_entry[field_name] = train_cover_acc
        else:
            log_entry[field_name] = "N/A"

        field_name = 'Val_Acc_cover'
        if val_cover_acc is not None:
            log_entry[field_name] = val_cover_acc
        else:
            log_entry[field_name] = "N/A"

        # 添加各隐写方法的准确率
        for method in sorted(all_methods):
            if method and method.lower() != 'cover':
                train_acc = train_tracker.get_cumulative_accuracy(method)
                val_acc = val_tracker.get_cumulative_accuracy(method)
                
                if train_acc is not None:
                    log_entry[f'Train_Acc_{method}'] = train_acc
                else:
                    log_entry[f'Train_Acc_{method}'] = "N/A"
                    
                if val_acc is not None:
                    log_entry[f'Val_Acc_{method}'] = val_acc
                else:
                    log_entry[f'Val_Acc_{method}'] = "N/A"

        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(log_entry)
            csvfile.flush()


        # 打印当前epoch的训练和验证结果
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/train_total:.4f}, Train Acc: {100.*train_correct/train_total:.2f}%, Val Loss: {val_loss/val_total:.4f}, Val Acc: {100.*val_correct/val_total:.2f}%')
        print(f"Validation prediction distribution: Label 0 (Cover): {val_preds[0]}, Label 1 (Stego): {val_preds[1]}")
        print(f'Time taken: {time.time() - epoch_start_time:.2f} seconds')
        

        # 在每个验证阶段结束后更新学习率
        scheduler.step()
        
        if(epoch + 1) % 5 == 0:
            # 打印每种隐写方法的准确率
            print("\n训练集各隐写方法准确率:")
            train_tracker.print_stats(epoch=epoch+1, use_cumulative=True)
            print("\n验证集各隐写方法准确率:")
            val_tracker.print_stats(epoch=epoch+1, use_cumulative=True)
            print("-" * 80)

        # 更新最佳验证准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'outputs/best_model.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 绘制常规训练图表
    plt.figure(figsize=(12, 5))
    
    # 图1: 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 图2: 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    plt.savefig(f'training_results_{timestamp}.png', dpi=300)

    print(f"训练结果图表已保存至 training_results_{timestamp}.png")
        
    StegoMethodTracker.plot_combined_accuracy_curves(
        train_tracker, 
        val_tracker, 
        save_path=f'combined_accuracy_curves_{timestamp}.png'
    )
    
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
    os.makedirs('outputs', exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(
        config['DATA_CONFIG']['data_dir'],
        config['DATA_CONFIG']['batch_size'],
        config['DATA_CONFIG']['num_workers']
    )
    print("数据加载器创建完成")
    
    # 创建模型
    model = CNNStegAnalysis(
        input_channels=config['MODEL_CONFIG']['input_channels'], dropout_rate=0.6
    ).to(device)
    
    # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=config['MODEL_CONFIG']['learning_rate'])

    # class_weights = torch.FloatTensor([2, 1]).to(device)  # 根据经验设置适当值
    # print(f"使用预设类别权重: {class_weights}")

    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            
        def forward(self, inputs, targets):
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
            return F_loss.mean()

    # 替换原有损失函数        
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=config['MODEL_CONFIG']['learning_rate'], 
                        weight_decay=1e-3) 
    
    print("模型创建完成")
    
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


