# -*- coding: utf-8 -*-
"""
main.py
RNN/GRU/LSTM 三模型对比训练系统 - 主入口脚本
智慧城市时序预测任务

功能：
1. 同时训练 RNN、GRU、LSTM 三种模型
2. 支持北京PM2.5和加州交通两个数据集
3. 生成完整的可视化分析报告
4. 支持长期依赖能力测试
5. 支持断点续训

使用方法：
    python main.py --dataset beijing_pm25 --epochs 100
    python main.py --dataset california_traffic --epochs 100 --long_term_test
"""

import os
import sys
import argparse
import torch
import numpy as np

from datasets import get_dataloaders
from train_models import (
    MultiModelComparison,
    LongTermDependencyTest,
    ModelTrainer,
    SleepRNN,
)
from visualize_comparison import (
    ComparisonVisualizer,
    load_and_visualize,
)


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='RNN/GRU/LSTM 三模型对比训练系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练北京PM2.5数据集
  python main.py --dataset beijing_pm25 --epochs 100

  # 训练加州交通数据集
  python main.py --dataset california_traffic --epochs 100

  # 长期依赖能力测试
  python main.py --dataset beijing_pm25 --long_term_test

  # 仅生成可视化（基于已有训练结果）
  python main.py --dataset beijing_pm25 --visualize_only

  # 断点续训
  python main.py --dataset beijing_pm25 --resume
        """
    )

    # 数据集配置
    parser.add_argument(
        '--dataset', type=str, default='beijing_pm25',
        choices=['beijing_pm25', 'california_traffic'],
        help='选择数据集 (默认: beijing_pm25)'
    )
    parser.add_argument(
        '--data_dir', type=str, default='./data',
        help='数据保存目录 (默认: ./data)'
    )

    # 模型配置
    parser.add_argument(
        '--seq_length', type=int, default=24,
        help='输入序列长度 (默认: 24)'
    )
    parser.add_argument(
        '--pred_length', type=int, default=6,
        help='预测序列长度 (默认: 6)'
    )
    parser.add_argument(
        '--hidden_size', type=int, default=64,
        help='RNN隐藏层维度 (默认: 64)'
    )
    parser.add_argument(
        '--num_layers', type=int, default=2,
        help='RNN层数 (默认: 2)'
    )

    # 训练配置
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='训练轮数 (默认: 100)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='批量大小 (默认: 64)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='学习率 (默认: 0.001)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='从检查点恢复训练（断点续训）'
    )

    # 测试配置
    parser.add_argument(
        '--long_term_test', action='store_true',
        help='执行长期依赖能力测试'
    )
    parser.add_argument(
        '--seq_lengths', type=int, nargs='+', default=None,
        help='长期依赖测试的序列长度列表 (默认: 6 12 24 48 72 96)'
    )

    # 输出配置
    parser.add_argument(
        '--checkpoint_dir', type=str, default='./checkpoints',
        help='模型检查点保存目录 (默认: ./checkpoints)'
    )
    parser.add_argument(
        '--results_dir', type=str, default='./results',
        help='训练结果保存目录 (默认: ./results)'
    )
    parser.add_argument(
        '--figures_dir', type=str, default='./figures',
        help='可视化图表保存目录 (默认: ./figures)'
    )

    # 模式配置
    parser.add_argument(
        '--visualize_only', action='store_true',
        help='仅生成可视化，不重新训练'
    )
    parser.add_argument(
        '--no_visualize', action='store_true',
        help='训练后不生成可视化'
    )

    # 设备配置
    parser.add_argument(
        '--device', type=str, default=None,
        help='计算设备 (cuda/cpu，默认自动选择)'
    )

    return parser.parse_args()


def setup_device(args_device: str = None) -> torch.device:
    """
    设置计算设备

    Args:
        args_device: 命令行指定的设备

    Returns:
        torch.device 实例
    """
    if args_device:
        device = torch.device(args_device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print(f"[系统] 使用计算设备: {device}")
    if device.type == 'cuda':
        print(f"[系统] GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"[系统] 可用显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    return device


def train_models(args: argparse.Namespace, device: torch.device) -> MultiModelComparison:
    """
    训练三种模型

    Args:
        args: 命令行参数
        device: 计算设备

    Returns:
        MultiModelComparison 实例
    """
    print("\n" + "=" * 70)
    print("  阶段一：模型训练")
    print("=" * 70)

    # 创建对比训练系统
    comparison = MultiModelComparison(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        seq_length=args.seq_length,
        pred_length=args.pred_length,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        device=device,
    )

    # 训练所有模型
    records = comparison.train_all_models(
        resume=args.resume,
        convergence_threshold=0.01,
    )

    return comparison


def generate_visualizations(
    comparison: MultiModelComparison,
    args: argparse.Namespace,
) -> ComparisonVisualizer:
    """
    生成可视化分析

    Args:
        comparison: 多模型对比实例
        args: 命令行参数

    Returns:
        ComparisonVisualizer 实例
    """
    print("\n" + "=" * 70)
    print("  阶段二：可视化分析")
    print("=" * 70)

    # 创建可视化器
    visualizer = ComparisonVisualizer(
        records=comparison.records,
        dataset_name=args.dataset,
        output_dir=args.figures_dir,
    )

    # 获取测试集预测结果
    predictions_dict = {}
    targets = None
    for model_type in comparison.MODEL_TYPES:
        try:
            preds, targs = comparison.get_test_predictions(model_type)
            predictions_dict[model_type] = preds
            if targets is None:
                targets = targs
        except Exception as e:
            print(f"[警告] 无法获取 {model_type} 的预测结果: {e}")

    # 生成所有可视化
    visualizer.generate_all_visualizations(
        predictions_dict=predictions_dict,
        targets=targets,
    )

    return visualizer


def run_long_term_test(args: argparse.Namespace, device: torch.device):
    """
    运行长期依赖能力测试

    Args:
        args: 命令行参数
        device: 计算设备
    """
    print("\n" + "=" * 70)
    print("  阶段三：长期依赖能力测试")
    print("=" * 70)

    test = LongTermDependencyTest(
        dataset_name=args.dataset,
        seq_lengths=args.seq_lengths,
        data_dir=args.data_dir,
        pred_length=args.pred_length,
        num_epochs=min(args.epochs, 50),  # 长期测试使用较少epoch
        device=device,
    )

    results = test.run()

    # 可视化长期依赖测试结果
    for metric in ['mse', 'mae', 'rmse']:
        ComparisonVisualizer.plot_long_term_dependency_results(
            results=results,
            dataset_name=args.dataset,
            output_dir=args.figures_dir,
            metric=metric,
            save=True,
        )

    return results


def print_summary(args: argparse.Namespace):
    """
    打印系统配置摘要
    """
    print("\n" + "=" * 70)
    print("  RNN/GRU/LSTM 三模型对比训练系统")
    print("  智慧城市时序预测任务")
    print("=" * 70)
    print(f"\n[配置] 数据集: {args.dataset}")
    print(f"[配置] 序列长度: {args.seq_length}, 预测长度: {args.pred_length}")
    print(f"[配置] 隐藏层: {args.hidden_size}, 层数: {args.num_layers}")
    print(f"[配置] 训练轮数: {args.epochs}, 批量大小: {args.batch_size}")
    print(f"[配置] 学习率: {args.lr}")
    print(f"[配置] 断点续训: {'是' if args.resume else '否'}")
    print(f"[配置] 长期依赖测试: {'是' if args.long_term_test else '否'}")
    print(f"[配置] 仅可视化模式: {'是' if args.visualize_only else '否'}")
    print("=" * 70)


def main():
    """
    主函数
    """
    # 解析参数
    args = parse_arguments()

    # 打印配置摘要
    print_summary(args)

    # 设置设备
    device = setup_device(args.device)

    # 仅可视化模式
    if args.visualize_only:
        print("\n[模式] 仅生成可视化图表...")
        try:
            visualizer = load_and_visualize(
                dataset_name=args.dataset,
                results_dir=args.results_dir,
                output_dir=args.figures_dir,
            )
            visualizer.generate_all_visualizations()
        except FileNotFoundError as e:
            print(f"[错误] 未找到训练记录文件: {e}")
            print("[提示] 请先运行训练: python main.py --dataset " + args.dataset)
            sys.exit(1)
        return

    # 阶段一：训练模型
    comparison = train_models(args, device)

    # 阶段二：生成可视化
    if not args.no_visualize:
        generate_visualizations(comparison, args)

    # 阶段三：长期依赖能力测试
    if args.long_term_test:
        run_long_term_test(args, device)

    # 最终总结
    print("\n" + "=" * 70)
    print("  所有任务完成！")
    print("=" * 70)
    print(f"\n[输出] 模型检查点: {os.path.abspath(args.checkpoint_dir)}")
    print(f"[输出] 训练记录: {os.path.abspath(args.results_dir)}")
    print(f"[输出] 可视化图表: {os.path.abspath(args.figures_dir)}")
    print("\n生成的文件包括:")
    print("  - 训练收敛曲线图")
    print("  - 梯度范数分析图")
    print("  - 预测结果对比图")
    print("  - 性能对比表格 (CSV/Markdown)")
    print("  - 综合对比仪表盘")
    if args.long_term_test:
        print("  - 长期依赖能力测试图")
    print("=" * 70)


if __name__ == "__main__":
    main()
