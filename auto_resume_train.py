#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动检测断点运行脚本
Auto Resume Training Script

该脚本会自动检测是否存在训练断点，如果存在则自动恢复训练，否则开始新的训练。
This script automatically detects training checkpoints and resumes training if available, 
otherwise starts new training.

使用方法 (Usage):
python auto_resume_train.py --config ./configs/zoomnet/cod_zoomnet.py
python auto_resume_train.py --config ./configs/zoomnet/sod_zoomnet.py --model-name ZoomNet --batch-size 16
"""

import argparse
import os
import sys
import glob
import json
import subprocess
from pathlib import Path

def get_python_command():
    """
    检测并返回虚拟环境中的Python命令
    Detect and return Python command from virtual environment
    """
    venv_paths = [
        "./venv/Scripts/python.exe",
        "./env/Scripts/python.exe", 
        "./.venv/Scripts/python.exe",
        "./Scripts/python.exe",
        "./venv/bin/python",  # Linux/Mac
        "./env/bin/python",   # Linux/Mac
        "./.venv/bin/python", # Linux/Mac
    ]
    
    # 检查虚拟环境
    for venv_path in venv_paths:
        if os.path.exists(venv_path):
            try:
                # 测试Python命令是否可用
                result = subprocess.run([venv_path, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"找到虚拟环境 (Found virtual environment): {venv_path}")
                    print(f"Python版本 (Python version): {result.stdout.strip()}")
                    return venv_path
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                continue
    
    # 检查系统Python
    try:
        result = subprocess.run(["python", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("警告: 未找到虚拟环境，使用系统Python (Warning: No virtual environment found, using system Python)")
            print("建议创建虚拟环境 (Recommend creating virtual environment): python -m venv venv")
            print(f"Python版本 (Python version): {result.stdout.strip()}")
            return "python"
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    
    print("错误: 未找到可用的Python环境 (Error: No available Python environment found)")
    return None


def find_latest_checkpoint(output_dir, exp_name):
    """
    查找最新的checkpoint文件
    Find the latest checkpoint file
    
    Args:
        output_dir (str): 输出目录
        exp_name (str): 实验名称
    
    Returns:
        tuple: (checkpoint_path, epoch) 或 (None, 0)
    """
    checkpoint_dir = os.path.join(output_dir, exp_name, "pth")
    
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # 查找所有checkpoint文件
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None, 0
    
    # 提取epoch数字并找到最新的
    def extract_epoch(filename):
        try:
            basename = os.path.basename(filename)
            epoch_str = basename.replace("checkpoint_epoch_", "").replace(".pth", "")
            return int(epoch_str)
        except ValueError:
            return 0
    
    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    latest_epoch = extract_epoch(latest_checkpoint)
    
    return latest_checkpoint, latest_epoch


def construct_exp_name_from_config(config_path, model_name=None, batch_size=None):
    """
    从配置文件构造实验名称
    Construct experiment name from config file
    """
    # 简单实现，实际应该导入utils.misc.construct_exp_name
    # Simple implementation, should actually import utils.misc.construct_exp_name
    config_name = Path(config_path).stem
    exp_parts = [config_name]
    
    if model_name:
        exp_parts.append(model_name)
    if batch_size:
        exp_parts.append(f"BS{batch_size}")
    
    return "_".join(exp_parts)


def check_training_status(output_dir, exp_name):
    """
    检查训练状态
    Check training status
    
    Returns:
        dict: 包含训练状态信息的字典
    """
    exp_dir = os.path.join(output_dir, exp_name)
    log_file = os.path.join(exp_dir, "tr_2025-06-24.txt")  # 根据实际日志文件名调整
    
    status = {
        "exp_exists": os.path.exists(exp_dir),
        "has_logs": os.path.exists(log_file),
        "checkpoint_path": None,
        "latest_epoch": 0,
        "is_completed": False,
        "log_lines": 0
    }
    
    # 检查checkpoint
    checkpoint_path, latest_epoch = find_latest_checkpoint(output_dir, exp_name)
    status["checkpoint_path"] = checkpoint_path
    status["latest_epoch"] = latest_epoch
      # 检查日志
    if status["has_logs"]:
        try:
            # 尝试不同的编码方式读取日志文件
            encodings = ['utf-8', 'gbk', 'latin-1']
            lines = []
            for encoding in encodings:
                try:
                    with open(log_file, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            
            status["log_lines"] = len(lines)
            # 检查是否有"End training..."标记
            if any("End training..." in line for line in lines):
                status["is_completed"] = True
        except Exception as e:
            print(f"Warning: Could not read log file {log_file}: {e}")
    
    return status


def main():
    parser = argparse.ArgumentParser("Auto Resume Training Script")
    parser.add_argument("--config", default="./configs/zoomnet/cod_zoomnet.py", type=str, 
                       help="配置文件路径 (Path to config file)")
    parser.add_argument("--datasets-info", default="./configs/_base_/dataset/dataset_configs.json", type=str)
    parser.add_argument("--model-name", type=str, help="模型名称 (Model name)")
    parser.add_argument("--batch-size", type=int, help="批次大小 (Batch size)")
    parser.add_argument("--info", type=str, help="实验标签 (Experiment tag)")
    parser.add_argument("--force-restart", action="store_true", 
                       help="强制重新开始训练，忽略现有checkpoint (Force restart training, ignore existing checkpoints)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="只检查状态，不执行训练 (Only check status, don't run training)")
    
    args = parser.parse_args()
    
    # 检查Python环境
    python_cmd = get_python_command()
    if not python_cmd:
        return 1
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在 (Error: Config file not found): {args.config}")
        return 1
    
    # 构造实验名称 (这里需要实际的构造逻辑)
    try:
        # 尝试导入实际的构造函数
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from utils import misc, configurator
        
        config = configurator.Configurator.fromfile(args.config)
        if args.model_name:
            config.model_name = args.model_name
        if args.batch_size:
            config.train.batch_size = args.batch_size
        if args.info:
            config.experiment_tag = args.info
            
        exp_name = misc.construct_exp_name(model_name=config.model_name, cfg=config.__dict__)
        output_dir = "./output"
        
    except Exception as e:
        print(f"警告: 无法导入配置工具，使用简化版本 (Warning: Cannot import config tools, using simplified version): {e}")
        exp_name = construct_exp_name_from_config(args.config, args.model_name, args.batch_size)
        output_dir = "./output"
    
    print(f"实验名称 (Experiment name): {exp_name}")
    print(f"输出目录 (Output directory): {output_dir}")
    
    # 检查训练状态
    status = check_training_status(output_dir, exp_name)
    
    print("\n=== 训练状态检查 (Training Status Check) ===")
    print(f"实验目录存在 (Experiment directory exists): {status['exp_exists']}")
    print(f"日志文件存在 (Log file exists): {status['has_logs']}")
    print(f"日志行数 (Log lines): {status['log_lines']}")
    print(f"最新checkpoint (Latest checkpoint): {status['checkpoint_path']}")
    print(f"最新epoch (Latest epoch): {status['latest_epoch']}")
    print(f"训练已完成 (Training completed): {status['is_completed']}")
    
    if args.dry_run:
        print("\n=== DRY RUN 模式，不执行训练 (DRY RUN mode, not executing training) ===")
        return 0
      # 决定训练策略
    train_command = [python_cmd, "main.py", "--config", args.config]
    
    if args.datasets_info:
        train_command.extend(["--datasets-info", args.datasets_info])
    if args.model_name:
        train_command.extend(["--model-name", args.model_name])
    if args.batch_size:
        train_command.extend(["--batch-size", str(args.batch_size)])
    if args.info:
        train_command.extend(["--info", args.info])
    
    if status['is_completed']:
        print("\n=== 训练已完成 (Training already completed) ===")
        print("如果要重新训练，请使用 --force-restart 参数")
        print("If you want to retrain, use --force-restart parameter")
        return 0
    
    elif status['checkpoint_path'] and not args.force_restart:
        print(f"\n=== 发现checkpoint，恢复训练 (Found checkpoint, resuming training) ===")
        print(f"从epoch {status['latest_epoch']} 恢复训练")
        print(f"Resuming training from epoch {status['latest_epoch']}")
        train_command.extend(["--resume-from", status['checkpoint_path']])
    
    elif args.force_restart and status['exp_exists']:
        print(f"\n=== 强制重新开始训练 (Force restart training) ===")
        print("警告: 现有的训练结果将被覆盖!")
        print("Warning: Existing training results will be overwritten!")
        
        import shutil
        exp_dir = os.path.join(output_dir, exp_name)
        backup_dir = f"{exp_dir}_backup_{os.getpid()}"
        print(f"备份现有实验到 (Backing up existing experiment to): {backup_dir}")
        try:
            shutil.move(exp_dir, backup_dir)
        except Exception as e:
            print(f"备份失败 (Backup failed): {e}")
    
    else:
        print(f"\n=== 开始新的训练 (Starting new training) ===")
      # 执行训练命令
    print(f"\n执行命令 (Executing command): {' '.join(train_command)}")
    
    try:
        result = subprocess.run(train_command, check=True)
        print("\n=== 训练完成 (Training completed) ===")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n=== 训练失败 (Training failed) ===")
        print(f"错误代码 (Error code): {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n=== 训练被用户中断 (Training interrupted by user) ===")
        return 130
    except Exception as e:
        print(f"\n=== 发生错误 (Error occurred) ===")
        print(f"错误信息 (Error message): {e}")
        return 1


if __name__ == "__main__":
    exit(main())
