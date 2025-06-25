#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练日志可视化脚本 (Training Log Visualization Script)

该脚本解析ZoomNet训练日志文件，提取损失和学习率数据，并绘制折线图。
This script parses ZoomNet training log files, extracts loss and learning rate data, and plots line charts.

使用方法 (Usage):
python plot_training_log.py --log-file ./output/experiment/tr_2025-06-25.txt
python plot_training_log.py --log-file ./output/experiment/tr_2025-06-25.txt --save-path ./plots/
"""

import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import csv

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_log_line(line):
    """
    解析单行日志数据
    Parse single log line data
    
    Expected format:
    I:3160:30000 160/750 4/40 | Lr:0.001054 | M:0.14308/C:0.15306 | [8, 1, 384, 384] | seg_BCE: 0.14933 seg_UAL_0.02713: 0.00373
    """
    # 正则表达式匹配日志格式
    pattern = r'I:(\d+):(\d+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)\s+\|\s+Lr:([\d.]+)\s+\|\s+M:([\d.]+)/C:([\d.]+)\s+\|\s+.*?\|\s+seg_BCE:\s+([\d.]+)\s+seg_UAL_[\d.]+:\s+([\d.]+)'
    
    match = re.match(pattern, line.strip())
    if match:
        return {
            'iteration': int(match.group(1)),
            'total_iterations': int(match.group(2)),
            'batch_idx': int(match.group(3)),
            'epoch_length': int(match.group(4)),
            'epoch': int(match.group(5)),
            'total_epochs': int(match.group(6)),
            'learning_rate': float(match.group(7)),
            'mean_loss': float(match.group(8)),
            'current_loss': float(match.group(9)),
            'seg_bce_loss': float(match.group(10)),
            'seg_ual_loss': float(match.group(11))
        }
    return None

def parse_training_log(log_file_path, encoding='utf-8'):
    """
    解析训练日志文件
    Parse training log file
    """
    data = []
    
    # 尝试不同的编码格式
    encodings = [encoding, 'utf-8', 'gbk', 'gb2312', 'latin-1']
    
    for enc in encodings:
        try:
            with open(log_file_path, 'r', encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法用任何编码格式读取文件: {log_file_path}")
    
    print(f"成功读取日志文件，共 {len(lines)} 行")
    
    # 解析每一行
    parsed_count = 0
    for line_num, line in enumerate(lines, 1):
        if line.strip() and line.startswith('I:'):
            parsed_data = parse_log_line(line)
            if parsed_data:
                data.append(parsed_data)
                parsed_count += 1
    
    print(f"成功解析 {parsed_count} 行训练数据")
    
    if not data:
        raise ValueError("未找到有效的训练数据行")
    
    return data

def plot_training_curves(data, save_path=None, show_plots=True):
    """
    绘制训练曲线
    Plot training curves
    """
    # 提取数据
    iterations = [d['iteration'] for d in data]
    mean_losses = [d['mean_loss'] for d in data]
    current_losses = [d['current_loss'] for d in data]
    learning_rates = [d['learning_rate'] for d in data]
    seg_bce_losses = [d['seg_bce_loss'] for d in data]
    seg_ual_losses = [d['seg_ual_loss'] for d in data]
    epochs = [d['epoch'] for d in data]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ZoomNet 训练过程可视化 (Training Process Visualization)', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线 (Loss Curves)
    ax1 = axes[0, 0]
    ax1.plot(iterations, mean_losses, 'b-', label='平均损失 (Mean Loss)', linewidth=1.5, alpha=0.8)
    ax1.plot(iterations, current_losses, 'r-', label='当前损失 (Current Loss)', linewidth=1, alpha=0.6)
    ax1.set_xlabel('迭代次数 (Iteration)')
    ax1.set_ylabel('损失值 (Loss)')
    ax1.set_title('训练损失曲线 (Training Loss Curve)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 学习率曲线 (Learning Rate Curve)
    ax2 = axes[0, 1]
    ax2.plot(iterations, learning_rates, 'g-', label='学习率 (Learning Rate)', linewidth=2)
    ax2.set_xlabel('迭代次数 (Iteration)')
    ax2.set_ylabel('学习率 (Learning Rate)')
    ax2.set_title('学习率变化曲线 (Learning Rate Schedule)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 3. 分项损失曲线 (Component Loss Curves)
    ax3 = axes[1, 0]
    ax3.plot(iterations, seg_bce_losses, 'm-', label='BCE损失 (BCE Loss)', linewidth=1.5)
    ax3.plot(iterations, seg_ual_losses, 'c-', label='UAL损失 (UAL Loss)', linewidth=1.5)
    ax3.set_xlabel('迭代次数 (Iteration)')
    ax3.set_ylabel('损失值 (Loss)')
    ax3.set_title('分项损失曲线 (Component Loss Curves)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 按Epoch分组的平均损失 (Average Loss per Epoch)
    ax4 = axes[1, 1]
    
    # 计算每个epoch的平均损失
    epoch_losses = defaultdict(list)
    epoch_bce_losses = defaultdict(list)
    
    for d in data:
        epoch_losses[d['epoch']].append(d['mean_loss'])
        epoch_bce_losses[d['epoch']].append(d['seg_bce_loss'])
    
    epoch_nums = sorted(epoch_losses.keys())
    avg_losses = [np.mean(epoch_losses[e]) for e in epoch_nums]
    avg_bce_losses = [np.mean(epoch_bce_losses[e]) for e in epoch_nums]
    
    ax4.plot(epoch_nums, avg_losses, 'o-', label='平均损失 (Mean Loss)', linewidth=2, markersize=4)
    ax4.plot(epoch_nums, avg_bce_losses, 's-', label='BCE损失 (BCE Loss)', linewidth=2, markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('平均损失值 (Average Loss)')
    ax4.set_title('每个Epoch的平均损失 (Average Loss per Epoch)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(save_path, f"training_curves_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {plot_file}")
        
        # 也保存为PDF格式
        pdf_file = os.path.join(save_path, f"training_curves_{timestamp}.pdf")
        plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
        print(f"PDF已保存到: {pdf_file}")
    
    # 显示图片
    if show_plots:
        plt.show()
    
    return fig

def plot_detailed_analysis(data, save_path=None, show_plots=True):
    """
    绘制详细分析图表
    Plot detailed analysis charts
    """
    # 提取数据
    iterations = [d['iteration'] for d in data]
    mean_losses = [d['mean_loss'] for d in data]
    learning_rates = [d['learning_rate'] for d in data]
    epochs = [d['epoch'] for d in data]
    total_iterations = data[0]['total_iterations']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('ZoomNet 训练详细分析 (Detailed Training Analysis)', fontsize=16, fontweight='bold')
    
    # 1. 损失分布直方图 (Loss Distribution)
    ax1 = axes[0, 0]
    ax1.hist(mean_losses, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('平均损失值 (Mean Loss)')
    ax1.set_ylabel('频次 (Frequency)')
    ax1.set_title('损失分布直方图 (Loss Distribution)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 学习率 vs 损失散点图 (Learning Rate vs Loss)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(learning_rates, mean_losses, 
                         c=iterations, cmap='viridis', alpha=0.6, s=20)
    ax2.set_xlabel('学习率 (Learning Rate)')
    ax2.set_ylabel('平均损失 (Mean Loss)')
    ax2.set_title('学习率与损失关系 (LR vs Loss)')
    ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    plt.colorbar(scatter, ax=ax2, label='迭代次数 (Iteration)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 损失平滑曲线 (Smoothed Loss)
    ax3 = axes[0, 2]
    window_size = max(1, len(data) // 50)  # 动态窗口大小
    
    # 计算移动平均
    smoothed_loss = []
    for i in range(len(mean_losses)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(mean_losses), i + window_size // 2 + 1)
        smoothed_loss.append(np.mean(mean_losses[start_idx:end_idx]))
    
    ax3.plot(iterations, mean_losses, 'lightblue', alpha=0.5, label='原始损失 (Raw Loss)')
    ax3.plot(iterations, smoothed_loss, 'darkblue', linewidth=2, label=f'平滑损失 (Smoothed, window={window_size})')
    ax3.set_xlabel('迭代次数 (Iteration)')
    ax3.set_ylabel('损失值 (Loss)')
    ax3.set_title('损失平滑曲线 (Smoothed Loss Curve)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 训练进度 (Training Progress)
    ax4 = axes[1, 0]
    progress = [iter_num / total_iterations * 100 for iter_num in iterations]
    ax4.plot(iterations, progress, 'orange', linewidth=2)
    ax4.set_xlabel('迭代次数 (Iteration)')
    ax4.set_ylabel('训练进度 (Progress %)')
    ax4.set_title('训练进度 (Training Progress)')
    ax4.grid(True, alpha=0.3)
    
    # 5. 损失变化率 (Loss Change Rate)
    ax5 = axes[1, 1]
    loss_diff = [mean_losses[i] - mean_losses[i-1] if i > 0 else 0 for i in range(len(mean_losses))]
    ax5.plot(iterations[1:], loss_diff[1:], 'red', alpha=0.7, linewidth=1)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('迭代次数 (Iteration)')
    ax5.set_ylabel('损失变化率 (Loss Change)')
    ax5.set_title('损失变化率 (Loss Change Rate)')
    ax5.grid(True, alpha=0.3)
    
    # 6. 每个Epoch的统计信息 (Epoch Statistics)
    ax6 = axes[1, 2]
    
    # 计算每个epoch的统计
    epoch_stats = defaultdict(list)
    for d in data:
        epoch_stats[d['epoch']].append(d['mean_loss'])
    
    epoch_nums = sorted(epoch_stats.keys())
    epoch_means = [np.mean(epoch_stats[e]) for e in epoch_nums]
    epoch_stds = [np.std(epoch_stats[e]) for e in epoch_nums]
    
    ax6.errorbar(epoch_nums, epoch_means, yerr=epoch_stds, 
                fmt='o-', capsize=5, capthick=2)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('损失值 (Loss)')
    ax6.set_title('每个Epoch的损失统计 (Loss Statistics per Epoch)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(save_path, f"detailed_analysis_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"详细分析图已保存到: {plot_file}")
    
    # 显示图片
    if show_plots:
        plt.show()
    
    return fig

def save_data_to_csv(data, save_path):
    """
    保存训练数据到CSV文件
    Save training data to CSV file
    """
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(save_path, f"training_data_{timestamp}.csv")
    
    # 写入CSV文件
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if data:
            # 获取字段名
            fieldnames = data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据
            for row in data:
                writer.writerow(row)
    
    print(f"训练数据已保存到CSV: {csv_file}")

def print_training_summary(data):
    """
    打印训练摘要信息
    Print training summary
    """
    if not data:
        return
        
    mean_losses = [d['mean_loss'] for d in data]
    learning_rates = [d['learning_rate'] for d in data]
    
    print("\n" + "="*60)
    print("                训练摘要 (Training Summary)")
    print("="*60)
    
    # 基本信息
    print(f"总迭代次数 (Total Iterations): {data[-1]['iteration']:,}")
    print(f"总Epoch数 (Total Epochs): {data[-1]['epoch'] + 1}")
    print(f"数据点数量 (Data Points): {len(data):,}")
    
    # 损失统计
    print(f"\n损失统计 (Loss Statistics):")
    print(f"  初始损失 (Initial Loss): {mean_losses[0]:.6f}")
    print(f"  最终损失 (Final Loss): {mean_losses[-1]:.6f}")
    print(f"  最小损失 (Min Loss): {min(mean_losses):.6f}")
    print(f"  最大损失 (Max Loss): {max(mean_losses):.6f}")
    print(f"  损失改善 (Loss Improvement): {(mean_losses[0] - mean_losses[-1]):.6f}")
    print(f"  改善百分比 (Improvement %): {((mean_losses[0] - mean_losses[-1]) / mean_losses[0] * 100):.2f}%")
    
    # 学习率统计
    print(f"\n学习率统计 (Learning Rate Statistics):")
    print(f"  初始学习率 (Initial LR): {learning_rates[0]:.6f}")
    print(f"  最终学习率 (Final LR): {learning_rates[-1]:.6f}")
    print(f"  最大学习率 (Max LR): {max(learning_rates):.6f}")
    print(f"  最小学习率 (Min LR): {min(learning_rates):.6f}")
    
    # 训练稳定性
    recent_losses = mean_losses[-100:] if len(mean_losses) >= 100 else mean_losses  # 最近100个点
    print(f"\n训练稳定性 (Training Stability - Last {len(recent_losses)} points):")
    print(f"  损失标准差 (Loss Std): {np.std(recent_losses):.6f}")
    print(f"  损失变化系数 (Loss CV): {(np.std(recent_losses) / np.mean(recent_losses)):.4f}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser("Training Log Visualization Script")
    parser.add_argument("--log-file", type=str, required=True,
                       help="训练日志文件路径 (Path to training log file)")
    parser.add_argument("--save-path", type=str, default="./plots",
                       help="图片保存路径 (Path to save plots)")
    parser.add_argument("--encoding", type=str, default="utf-8",
                       help="日志文件编码 (Log file encoding)")
    parser.add_argument("--no-show", action="store_true",
                       help="不显示图片，只保存 (Don't show plots, save only)")
    parser.add_argument("--detailed", action="store_true",
                       help="生成详细分析图表 (Generate detailed analysis charts)")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.log_file):
        print(f"错误: 日志文件不存在 (Error: Log file not found): {args.log_file}")
        return 1
    
    try:
        # 解析日志文件
        print(f"正在解析日志文件: {args.log_file}")
        data = parse_training_log(args.log_file, args.encoding)
        
        # 打印摘要信息
        print_training_summary(data)
        
        # 绘制基本训练曲线
        print(f"\n正在生成训练曲线图...")
        plot_training_curves(data, args.save_path, not args.no_show)
        
        # 生成详细分析图表
        if args.detailed:
            print(f"正在生成详细分析图表...")
            plot_detailed_analysis(data, args.save_path, not args.no_show)
        
        # 保存数据到CSV
        if args.save_path:
            save_data_to_csv(data, args.save_path)
        
        print(f"\n可视化完成! (Visualization completed!)")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
