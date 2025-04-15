import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import time

def draw_training_plots(csv_file, output_dir='figures', dpi=300):
    """
    从训练日志CSV文件绘制训练结果图表
    
    参数:
    - csv_file: CSV文件路径
    - output_dir: 输出图片的目录
    - dpi: 输出图片的分辨率
    """

    # 输出文件夹加上timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)

    # 读取CSV文件，手动处理注释行
    with open(csv_file, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('//')]
    
    # 使用StringIO将过滤后的内容传给pandas
    from io import StringIO
    df = pd.read_csv(StringIO(''.join(lines)))
    
    # 替换"N/A"为NaN
    df = df.replace('N/A', np.nan)
    
    # 创建保存图片的目录(如果不存在)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 绘制Loss和Accuracy曲线
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss曲线
    axes[0].plot(df['Epoch'], df['Train_Loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].plot(df['Epoch'], df['Val_Loss'], 'r-', linewidth=2, label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy曲线
    axes[1].plot(df['Epoch'], df['Train_Acc'], 'b-', linewidth=2, label='Train Accuracy')
    axes[1].plot(df['Epoch'], df['Val_Acc'], 'r-', linewidth=2, label='Val Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_accuracy.png', dpi=dpi)
    plt.close()
    
    # 2. 绘制Cover图
    if 'Train_Acc_cover' in df.columns and 'Val_Acc_cover' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Train_Acc_cover'], 'b-', linewidth=2, label='Train Accuracy')
        plt.plot(df['Epoch'], df['Val_Acc_cover'], 'r-', linewidth=2, label='Val Accuracy')
        plt.title('Cover Detection Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cover_accuracy.png', dpi=dpi)
        plt.close()
    
    # 3. 计算并绘制LSBEE图
    lsbee_train_cols = [col for col in df.columns if col.startswith('Train_Acc_lsbee_')]
    lsbee_val_cols = [col for col in df.columns if col.startswith('Val_Acc_lsbee_')]
    
    if lsbee_train_cols and lsbee_val_cols:
        # 计算平均值
        df['Train_Acc_lsbee_avg'] = df[lsbee_train_cols].mean(axis=1)
        df['Val_Acc_lsbee_avg'] = df[lsbee_val_cols].mean(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Train_Acc_lsbee_avg'], 'b-', linewidth=2, label='Train Accuracy')
        plt.plot(df['Epoch'], df['Val_Acc_lsbee_avg'], 'r-', linewidth=2, label='Val Accuracy')
        plt.title('LSBEE Detection Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/lsbee_accuracy.png', dpi=dpi)
        plt.close()
    
    # 4. 计算并绘制MIN图
    min_train_cols = [col for col in df.columns if col.startswith('Train_Acc_min_')]
    min_val_cols = [col for col in df.columns if col.startswith('Val_Acc_min_')]
    
    if min_train_cols and min_val_cols:
        # 计算平均值
        df['Train_Acc_min_avg'] = df[min_train_cols].mean(axis=1)
        df['Val_Acc_min_avg'] = df[min_val_cols].mean(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Train_Acc_min_avg'], 'b-', linewidth=2, label='Train Accuracy')
        plt.plot(df['Epoch'], df['Val_Acc_min_avg'], 'r-', linewidth=2, label='Val Accuracy')
        plt.title('MIN Detection Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/min_accuracy.png', dpi=dpi)
        plt.close()
    
    # 5. 计算并绘制SIGN图
    sign_train_cols = [col for col in df.columns if col.startswith('Train_Acc_sign_')]
    sign_val_cols = [col for col in df.columns if col.startswith('Val_Acc_sign_')]
    
    if sign_train_cols and sign_val_cols:
        # 计算平均值
        df['Train_Acc_sign_avg'] = df[sign_train_cols].mean(axis=1)
        df['Val_Acc_sign_avg'] = df[sign_val_cols].mean(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Train_Acc_sign_avg'], 'b-', linewidth=2, label='Train Accuracy')
        plt.plot(df['Epoch'], df['Val_Acc_sign_avg'], 'r-', linewidth=2, label='Val Accuracy')
        plt.title('SIGN Detection Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sign_accuracy.png', dpi=dpi)
        plt.close()
    
    print(f"所有图表已保存到{output_dir}目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制训练结果图表')
    parser.add_argument('--csv_file', type=str, default="training_log_2025-04-14-22-25.csv", 
                        help='CSV日志文件路径')
    parser.add_argument('--output_dir', type=str, default="figures", 
                        help='输出图片的目录')
    parser.add_argument('--dpi', type=int, default=300, 
                        help='输出图片的DPI')
    
    args = parser.parse_args()
    draw_training_plots(args.csv_file, args.output_dir, args.dpi)