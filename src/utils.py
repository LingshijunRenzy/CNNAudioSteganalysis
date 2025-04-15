import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class StegoMethodTracker:
    """用于跟踪不同隐写方法性能指标的工具类，基于累积统计"""
    
    def __init__(self):
        # 为每种隐写方法初始化累积性能指标
        self.method_stats = defaultdict(lambda: {
            'correct': 0,          # 累积正确预测数
            'total': 0,            # 累积总样本数
            'epoch_correct': 0,    # 当前epoch的正确预测数
            'epoch_total': 0,      # 当前epoch的总样本数
            'epoch_accs': []       # 每个epoch的准确率
        })
        self.cover_stats = {
            'correct': 0,          # 累积正确预测数
            'total': 0,            # 累积总样本数
            'epoch_correct': 0,    # 当前epoch的正确预测数
            'epoch_total': 0,      # 当前epoch的总样本数
            'epoch_accs': []       # 每个epoch的准确率
        }
        self.all_methods = set()   # 所有发现的隐写方法
        self.epoch_counter = 0     # 当前epoch计数
        
    def update(self, predictions, labels, stego_methods):
        """
        更新每种隐写方法的统计信息
        
        Args:
            predictions: 模型预测结果
            labels: 真实标签
            stego_methods: 每个样本对应的隐写方法
        """
        for pred, label, method in zip(predictions, labels, stego_methods):
            correct = (pred == label)
            
            if method == "cover":  # cover音频
                self.cover_stats['correct'] += int(correct)  # 累积正确计数
                self.cover_stats['total'] += 1                # 累积总样本计数
                self.cover_stats['epoch_correct'] += int(correct)  # 当前epoch正确计数
                self.cover_stats['epoch_total'] += 1               # 当前epoch总样本计数
            else:  # stego音频
                self.method_stats[method]['correct'] += int(correct)  # 累积正确计数
                self.method_stats[method]['total'] += 1               # 累积总样本计数
                self.method_stats[method]['epoch_correct'] += int(correct)  # 当前epoch正确计数  
                self.method_stats[method]['epoch_total'] += 1               # 当前epoch总样本计数
                self.all_methods.add(method)
    
    def compute_epoch_accuracy(self):
        """计算并存储本轮epoch的准确率，但保留累积统计"""
        self.epoch_counter += 1
        
        # 计算cover准确率（当前epoch）
        if self.cover_stats['epoch_total'] > 0:
            acc = 100.0 * self.cover_stats['epoch_correct'] / self.cover_stats['epoch_total']
            self.cover_stats['epoch_accs'].append(acc)
        else:
            # 如果没有cover样本，添加一个None值保持列表长度一致
            self.cover_stats['epoch_accs'].append(None)
        
        # 重置当前epoch的计数器，但保留累积计数器
        self.cover_stats['epoch_correct'] = 0
        self.cover_stats['epoch_total'] = 0
        
        # 计算每种隐写方法的准确率（当前epoch）
        for method in self.all_methods:
            stats = self.method_stats[method]
            if stats['epoch_total'] > 0:
                acc = 100.0 * stats['epoch_correct'] / stats['epoch_total']
                stats['epoch_accs'].append(acc)
            else:
                # 如果没有特定方法的样本，添加None保持列表长度一致
                stats['epoch_accs'].append(None)
            
            # 重置当前epoch的计数器，但保留累积计数器
            stats['epoch_correct'] = 0
            stats['epoch_total'] = 0
    
    def get_cumulative_accuracy(self, method=None):
        """
        获取累积准确率（所有epoch的累积统计）
        
        Args:
            method: 隐写方法名，如果为None则返回cover的累积准确率
            
        Returns:
            累积准确率，如果没有样本则返回None
        """
        if method is None:
            # 获取cover的累积准确率
            if self.cover_stats['total'] > 0:
                return 100.0 * self.cover_stats['correct'] / self.cover_stats['total']
            return None
        else:
            # 获取特定方法的累积准确率
            if method in self.method_stats and self.method_stats[method]['total'] > 0:
                return 100.0 * self.method_stats[method]['correct'] / self.method_stats[method]['total']
            return None
    
    def print_stats(self, epoch=None, use_cumulative=True):
        """
        打印每种隐写方法的准确率
        
        Args:
            epoch: 当前epoch号，如果提供则在输出中显示
            use_cumulative: 是否使用累积统计的准确率，默认为True
        """
        if epoch is not None:
            print(f"Epoch {epoch} - 各隐写方法准确率:")
        else:
            print("各隐写方法准确率:")
        
        if use_cumulative:
            # 使用累积统计的准确率
            cover_acc = self.get_cumulative_accuracy()
            if cover_acc is not None:
                print(f"Cover: {cover_acc:.2f}% ({self.cover_stats['total']} 样本)")
            else:
                print("Cover: 无样本")
            
            for method in sorted(self.all_methods):
                acc = self.get_cumulative_accuracy(method)
                if acc is not None:
                    print(f"{method}: {acc:.2f}% ({self.method_stats[method]['total']} 样本)")
                else:
                    print(f"{method}: 无样本")
        else:
            # 使用当前epoch的准确率（与之前的行为一致）
            if self.cover_stats['epoch_accs'] and self.cover_stats['epoch_accs'][-1] is not None:
                print(f"Cover: {self.cover_stats['epoch_accs'][-1]:.2f}%")
            else:
                print("Cover: 无样本或未检测")
            
            for method in sorted(self.all_methods):
                stats = self.method_stats[method]
                if stats['epoch_accs'] and stats['epoch_accs'][-1] is not None:
                    print(f"{method}: {stats['epoch_accs'][-1]:.2f}%")
                else:
                    print(f"{method}: 无样本或未检测")
    
    def plot_accuracy_curves(self, save_path='stego_methods_accuracy.png', include_cumulative=True):
        """
        Plot accuracy curves for each steganography method
        
        Args:
            save_path: Path to save the chart
            include_cumulative: Whether to include cumulative accuracy in the chart
        """
        # Filter out None values for plotting
        cover_accs = [acc for acc in self.cover_stats['epoch_accs'] if acc is not None]
        epochs = range(1, len(cover_accs) + 1)
        
        if not epochs:  # If no valid data, return
            print("Not enough data to plot accuracy curves")
            return
        
        plt.figure(figsize=(14, 10))
        
        # Only plot cover data if available
        if cover_accs:
            plt.plot(epochs, cover_accs, 'k-', linewidth=2, label='Cover (Single Epoch)')
            # Add horizontal line for cumulative accuracy
            if include_cumulative:
                cum_acc = self.get_cumulative_accuracy()
                if cum_acc is not None:
                    plt.axhline(y=cum_acc, color='k', linestyle='--', 
                                label=f'Cover (Cumulative): {cum_acc:.2f}%')
        
        # Plot curves for each steganography method
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.all_methods)))
        for i, method in enumerate(sorted(self.all_methods)):
            stats = self.method_stats[method]
            # Filter out None values
            method_accs = [acc for acc in stats['epoch_accs'] if acc is not None]
            color = colors[i % len(colors)]
            
            if method_accs:
                method_epochs = range(1, len(method_accs) + 1)
                plt.plot(method_epochs, method_accs, color=color, linewidth=1.5, 
                         label=f'{method} (Single Epoch)')
                
                # Add horizontal line for cumulative accuracy
                if include_cumulative:
                    cum_acc = self.get_cumulative_accuracy(method)
                    if cum_acc is not None:
                        plt.axhline(y=cum_acc, color=color, linestyle='--',
                                   label=f'{method} (Cumulative): {cum_acc:.2f}%')
        
        plt.title('Accuracy Curves for Different Steganography Methods')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legend on the right side
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy curves for different steganography methods saved to {save_path}")
        
        # Plot second chart with only cumulative statistics
        if include_cumulative:
            plt.figure(figsize=(10, 6))
            
            # Collect all methods and their cumulative accuracies
            methods = ['Cover'] + sorted(list(self.all_methods))
            accuracies = []
            
            # Get cumulative accuracy for each method
            for method in methods:
                if method == 'Cover':
                    acc = self.get_cumulative_accuracy()
                else:
                    acc = self.get_cumulative_accuracy(method)
                
                if acc is not None:
                    accuracies.append(acc)
                else:
                    accuracies.append(0)  # Use 0 for methods with no samples
            
            # Create bar chart
            bars = plt.bar(methods, accuracies, color='skyblue')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.2f}%', ha='center', va='bottom')
            
            plt.title('Cumulative Accuracy for Different Steganography Methods')
            plt.xlabel('Steganography Method')
            plt.ylabel('Accuracy (%)')
            plt.ylim(0, 105)  # Set y-axis limit, leave space for labels
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels to prevent overlap
            plt.tight_layout()
            plt.savefig(save_path.replace('.png', '_cumulative.png'), dpi=300)
            print(f"Cumulative accuracy chart for different steganography methods saved to {save_path.replace('.png', '_cumulative.png')}")
    
    def reset_cumulative_stats(self):
        """每个epoch开始时重置累积统计"""
        # 仅重置计数器而不是替换整个字典
        self.cover_stats['correct'] = 0
        self.cover_stats['total'] = 0
        self.cover_stats['epoch_correct'] = 0  
        self.cover_stats['epoch_total'] = 0
        
        # 为每个隐写方法重置计数器
        for method in self.all_methods:
            if method in self.method_stats:
                self.method_stats[method]['correct'] = 0
                self.method_stats[method]['total'] = 0
                self.method_stats[method]['epoch_correct'] = 0
                self.method_stats[method]['epoch_total'] = 0

    @staticmethod
    def plot_combined_accuracy_curves(train_tracker, val_tracker, save_path='combined_stego_methods_accuracy.png'):
        """
        将训练和验证的准确率曲线以及累积准确率图表合并到一个图中
        
        Args:
            train_tracker: 训练集的StegoMethodTracker实例
            val_tracker: 验证集的StegoMethodTracker实例
            save_path: 保存路径
        """
        plt.figure(figsize=(16, 16))
        
        # 上半部分：训练和验证的准确率曲线
        # 训练集曲线图 (左上)
        plt.subplot(2, 2, 1)
        
        # 处理训练集曲线数据
        train_cover_accs = [acc for acc in train_tracker.cover_stats['epoch_accs'] if acc is not None]
        train_epochs = range(1, len(train_cover_accs) + 1)
        
        if train_cover_accs:
            plt.plot(train_epochs, train_cover_accs, 'k-', linewidth=2, label='Cover (Train)')
            
            # 添加训练集累积准确率水平线
            train_cum_acc = train_tracker.get_cumulative_accuracy()
            if train_cum_acc is not None:
                plt.axhline(y=train_cum_acc, color='k', linestyle='--', 
                            label=f'Cover (Train Cumulative): {train_cum_acc:.2f}%')
        
        # 针对每种隐写方法绘制训练曲线
        colors = plt.cm.tab20(np.linspace(0, 1, len(train_tracker.all_methods)))
        for i, method in enumerate(sorted(train_tracker.all_methods)):
            stats = train_tracker.method_stats[method]
            method_accs = [acc for acc in stats['epoch_accs'] if acc is not None]
            
            if method_accs:
                method_epochs = range(1, len(method_accs) + 1)
                color = colors[i % len(colors)]
                plt.plot(method_epochs, method_accs, color=color, linewidth=1.5, 
                         label=f'{method} (Train)')
                
                # 添加累积准确率水平线
                cum_acc = train_tracker.get_cumulative_accuracy(method)
                if cum_acc is not None:
                    plt.axhline(y=cum_acc, color=color, linestyle='--',
                               label=f'{method} (Train Cumulative): {cum_acc:.2f}%')
        
        plt.title('Training Accuracy Curves by Method')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.grid(True)
        
        # 验证集曲线图 (右上)
        plt.subplot(2, 2, 2)
        
        # 处理验证集曲线数据
        val_cover_accs = [acc for acc in val_tracker.cover_stats['epoch_accs'] if acc is not None]
        val_epochs = range(1, len(val_cover_accs) + 1)
        
        if val_cover_accs:
            plt.plot(val_epochs, val_cover_accs, 'k-', linewidth=2, label='Cover (Val)')
            
            # 添加验证集累积准确率水平线
            val_cum_acc = val_tracker.get_cumulative_accuracy()
            if val_cum_acc is not None:
                plt.axhline(y=val_cum_acc, color='k', linestyle='--', 
                            label=f'Cover (Val Cumulative): {val_cum_acc:.2f}%')
        
        # 针对每种隐写方法绘制验证曲线
        for i, method in enumerate(sorted(val_tracker.all_methods)):
            stats = val_tracker.method_stats[method]
            method_accs = [acc for acc in stats['epoch_accs'] if acc is not None]
            
            if method_accs:
                method_epochs = range(1, len(method_accs) + 1)
                color = colors[i % len(colors)]
                plt.plot(method_epochs, method_accs, color=color, linewidth=1.5, 
                         label=f'{method} (Val)')
                
                # 添加累积准确率水平线
                cum_acc = val_tracker.get_cumulative_accuracy(method)
                if cum_acc is not None:
                    plt.axhline(y=cum_acc, color=color, linestyle='--',
                               label=f'{method} (Val Cumulative): {cum_acc:.2f}%')
        
        plt.title('Validation Accuracy Curves by Method')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.grid(True)
        
        # 下半部分：累积准确率柱状图
        # 训练集累积准确率 (左下)
        plt.subplot(2, 2, 3)
        
        # 收集训练集各方法累积准确率
        train_methods = ['Cover'] + sorted(list(train_tracker.all_methods))
        train_accuracies = []
        
        # 获取训练集各方法累积准确率
        for method in train_methods:
            if method == 'Cover':
                acc = train_tracker.get_cumulative_accuracy()
            else:
                acc = train_tracker.get_cumulative_accuracy(method)
            
            train_accuracies.append(acc if acc is not None else 0)
        
        # 创建训练集柱状图
        train_bars = plt.bar(train_methods, train_accuracies, color='skyblue')
        
        # 添加训练集数值标签
        for bar in train_bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
        
        plt.title('Training Cumulative Accuracy by Method')
        plt.xlabel('Steganography Method')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 105)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.grid(axis='y')
        
        # 验证集累积准确率 (右下)
        plt.subplot(2, 2, 4)
        
        # 收集验证集各方法累积准确率
        val_methods = ['Cover'] + sorted(list(val_tracker.all_methods))
        val_accuracies = []
        
        # 获取验证集各方法累积准确率
        for method in val_methods:
            if method == 'Cover':
                acc = val_tracker.get_cumulative_accuracy()
            else:
                acc = val_tracker.get_cumulative_accuracy(method)
            
            val_accuracies.append(acc if acc is not None else 0)
        
        # 创建验证集柱状图
        val_bars = plt.bar(val_methods, val_accuracies, color='salmon')
        
        # 添加验证集数值标签
        for bar in val_bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
        
        plt.title('Validation Cumulative Accuracy by Method')
        plt.xlabel('Steganography Method')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 105)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.grid(axis='y')
        
        # 保存整合图表
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined accuracy curves for different steganography methods saved to {save_path}")