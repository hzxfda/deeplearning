# -*- coding: utf-8 -*-
"""
visualize_comparison.py
RNN/GRU/LSTM 三模型对比可视化分析模块
用于智慧城市时序预测任务的结果分析和展示
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional
from train_models import TrainingRecord, MultiModelComparison

# 设置中文字体支持
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = True
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'


# =============================================================================
# 可视化分析器
# =============================================================================

class ComparisonVisualizer:
    """
    模型对比可视化分析器
    提供多种可视化图表用于分析RNN/GRU/LSTM的性能差异
    """

    # 模型颜色配置
    MODEL_COLORS = {
        'RNN': '#E74C3C',   # 红色
        'GRU': '#3498DB',   # 蓝色
        'LSTM': '#2ECC71',  # 绿色
    }

    def __init__(
        self,
        records: Dict[str, TrainingRecord],
        dataset_name: str,
        output_dir: str = "./figures",
    ):
        """
        初始化可视化分析器

        Args:
            records: 各模型的训练记录字典 {model_type: TrainingRecord}
            dataset_name: 数据集名称
            output_dir: 图表输出目录
        """
        self.records = records
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 获取模型类型列表
        self.model_types = sorted(records.keys())

    # -------------------------------------------------------------------------
    # 1. 训练收敛曲线对比
    # -------------------------------------------------------------------------

    def plot_training_curves(self, figsize: Tuple[int, int] = (14, 5), save: bool = True) -> plt.Figure:
        """
        绘制三种模型的训练收敛曲线对比

        包含：
        - 训练Loss曲线
        - 验证Loss曲线
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 训练Loss
        ax1 = axes[0]
        for model_type in self.model_types:
            record = self.records[model_type]
            epochs = record.epochs
            train_losses = record.train_losses
            color = self.MODEL_COLORS.get(model_type, '#000000')
            ax1.plot(epochs, train_losses, label=model_type, color=color, linewidth=2)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss (MSE)', fontsize=12)
        ax1.set_title('训练Loss收敛曲线对比', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 验证Loss
        ax2 = axes[1]
        for model_type in self.model_types:
            record = self.records[model_type]
            epochs = record.epochs
            val_losses = record.val_losses
            color = self.MODEL_COLORS.get(model_type, '#000000')
            ax2.plot(epochs, val_losses, label=model_type, color=color, linewidth=2)

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation Loss (MSE)', fontsize=12)
        ax2.set_title('验证Loss收敛曲线对比', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        fig.suptitle(
            f'{self.dataset_name} - 模型收敛速度对比',
            fontsize=16, fontweight='bold', y=1.02
        )
        plt.tight_layout()

        if save:
            save_path = os.path.join(
                self.output_dir,
                f'{self.dataset_name}_training_curves.png'
            )
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[可视化] 训练收敛曲线已保存: {save_path}")

        return fig

    # -------------------------------------------------------------------------
    # 2. 梯度范数变化曲线（梯度消散分析）
    # -------------------------------------------------------------------------

    def plot_gradient_norms(self, figsize: Tuple[int, int] = (14, 10), save: bool = True) -> plt.Figure:
        """
        绘制梯度范数变化曲线

        分析各层梯度范数随epoch的变化，展示LSTM/GRU如何缓解梯度消散
        分别绘制：
        - 各模型总体梯度范数
        - 分层梯度范数对比（第一层 vs 最后一层）
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # (1) 总体梯度范数
        ax1 = axes[0]
        for model_type in self.model_types:
            record = self.records[model_type]
            epochs = record.epochs

            # 计算每个epoch的总体梯度范数（所有参数的平均）
            total_norms = []
            for grad_dict in record.grad_norms_per_epoch:
                if grad_dict:
                    avg_norm = np.mean(list(grad_dict.values()))
                    total_norms.append(avg_norm)
                else:
                    total_norms.append(0)

            color = self.MODEL_COLORS.get(model_type, '#000000')
            ax1.plot(epochs, total_norms, label=model_type, color=color, linewidth=2)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Average Gradient Norm', fontsize=12)
        ax1.set_title('总体梯度范数变化（梯度消散分析）', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 添加说明文字
        ax1.text(
            0.02, 0.98,
            '注：梯度范数快速衰减表明梯度消散严重\n'
            'LSTM/GRU通过门控机制保持梯度稳定',
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        # (2) 分层梯度范数对比：第一层 vs 最后一层
        ax2 = axes[1]

        for model_type in self.model_types:
            record = self.records[model_type]
            epochs = record.epochs

            # 提取第一层和最后一层的梯度范数
            first_layer_norms = []
            last_layer_norms = []

            for grad_dict in record.grad_norms_per_epoch:
                if not grad_dict:
                    first_layer_norms.append(0)
                    last_layer_norms.append(0)
                    continue

                # 找到包含 'weight_ih_l0' 的键（第一层输入权重）
                first_keys = [k for k in grad_dict.keys() if 'l0' in k and 'weight' in k]
                # 找到包含最大层数的键（最后一层）
                max_layer = max(
                    [int(k.split('l')[1].split('_')[0]) for k in grad_dict.keys() if 'l' in k and 'weight' in k],
                    default=0
                )
                last_keys = [k for k in grad_dict.keys() if f'l{max_layer}' in k and 'weight' in k]

                first_norm = np.mean([grad_dict[k] for k in first_keys if k in grad_dict]) if first_keys else 0
                last_norm = np.mean([grad_dict[k] for k in last_keys if k in grad_dict]) if last_keys else 0

                first_layer_norms.append(first_norm)
                last_layer_norms.append(last_norm)

            color = self.MODEL_COLORS.get(model_type, '#000000')

            # 绘制第一层梯度范数（实线）
            ax2.plot(epochs, first_layer_norms, label=f'{model_type} (第一层)',
                    color=color, linewidth=2, linestyle='-')
            # 绘制最后一层梯度范数（虚线）
            ax2.plot(epochs, last_layer_norms, label=f'{model_type} (最后一层)',
                    color=color, linewidth=2, linestyle='--', alpha=0.7)

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Gradient Norm', fontsize=12)
        ax2.set_title('分层梯度范数对比（第一层实线 / 最后一层虚线）', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9, ncol=3)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        fig.suptitle(
            f'{self.dataset_name} - 梯度消散分析',
            fontsize=16, fontweight='bold', y=0.995
        )
        plt.tight_layout()

        if save:
            save_path = os.path.join(
                self.output_dir,
                f'{self.dataset_name}_gradient_norms.png'
            )
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[可视化] 梯度范数曲线已保存: {save_path}")

        return fig

    # -------------------------------------------------------------------------
    # 3. 预测结果对比图
    # -------------------------------------------------------------------------

    def plot_predictions(
        self,
        predictions_dict: Dict[str, np.ndarray],
        targets: np.ndarray,
        sample_idx: int = 0,
        pred_step: int = 0,
        num_samples: int = 200,
        figsize: Tuple[int, int] = (14, 6),
        save: bool = True,
    ) -> plt.Figure:
        """
        绘制预测结果对比图

        Args:
            predictions_dict: 各模型的预测结果 {model_type: predictions_array}
            targets: 真实值数组
            sample_idx: 样本索引
            pred_step: 预测步长索引（0表示第一步预测）
            num_samples: 显示的样本数量
            figsize: 图像尺寸
            save: 是否保存
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 确保样本数量不超出范围
        n_samples = min(num_samples, len(targets))
        indices = range(n_samples)

        # 真实值
        true_values = targets[:n_samples, pred_step, 0]

        # (1) 所有模型预测对比
        ax1 = axes[0]
        ax1.plot(indices, true_values, label='真实值', color='black',
                linewidth=2, linestyle='-', alpha=0.8)

        for model_type in self.model_types:
            if model_type in predictions_dict:
                preds = predictions_dict[model_type][:n_samples, pred_step, 0]
                color = self.MODEL_COLORS.get(model_type, '#000000')
                ax1.plot(indices, preds, label=f'{model_type}预测',
                        color=color, linewidth=1.5, alpha=0.8)

        ax1.set_xlabel('样本索引', fontsize=12)
        ax1.set_ylabel('预测值', fontsize=12)
        ax1.set_title(f'预测结果对比（第{pred_step+1}步预测）', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # (2) 预测误差对比
        ax2 = axes[1]
        for model_type in self.model_types:
            if model_type in predictions_dict:
                preds = predictions_dict[model_type][:n_samples, pred_step, 0]
                errors = np.abs(preds - true_values)
                color = self.MODEL_COLORS.get(model_type, '#000000')
                ax2.plot(indices, errors, label=f'{model_type}误差',
                        color=color, linewidth=1.5, alpha=0.7)

        ax2.set_xlabel('样本索引', fontsize=12)
        ax2.set_ylabel('绝对误差', fontsize=12)
        ax2.set_title('预测绝对误差对比', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f'{self.dataset_name} - 预测效果对比',
            fontsize=16, fontweight='bold', y=1.02
        )
        plt.tight_layout()

        if save:
            save_path = os.path.join(
                self.output_dir,
                f'{self.dataset_name}_predictions.png'
            )
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[可视化] 预测结果对比图已保存: {save_path}")

        return fig

    # -------------------------------------------------------------------------
    # 4. 模型性能对比表格
    # -------------------------------------------------------------------------

    def generate_performance_table(self, save: bool = True) -> pd.DataFrame:
        """
        生成模型性能对比表格

        包含指标：MSE、MAE、RMSE、训练时间、收敛epoch、最佳验证loss
        """
        data = []
        for model_type in self.model_types:
            record = self.records[model_type]
            data.append({
                '模型': model_type,
                'MSE': f"{record.final_mse:.6f}",
                'MAE': f"{record.final_mae:.6f}",
                'RMSE': f"{record.final_rmse:.6f}",
                '最佳验证Loss': f"{record.best_val_loss:.6f}",
                '最佳Epoch': record.best_epoch,
                '收敛Epoch': record.convergence_epoch if record.convergence_epoch != -1 else '未收敛',
                '训练时间(秒)': f"{record.total_train_time:.2f}",
            })

        df = pd.DataFrame(data)

        # 打印表格
        print("\n" + "=" * 80)
        print(f"{self.dataset_name} - 模型性能对比表")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        if save:
            # 保存为CSV
            csv_path = os.path.join(
                self.output_dir,
                f'{self.dataset_name}_performance_table.csv'
            )
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"[可视化] 性能对比表已保存: {csv_path}")

            # 保存为Markdown
            md_path = os.path.join(
                self.output_dir,
                f'{self.dataset_name}_performance_table.md'
            )
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {self.dataset_name} - 模型性能对比\n\n")
                f.write(df.to_markdown(index=False))
            print(f"[可视化] Markdown表格已保存: {md_path}")

        return df

    # -------------------------------------------------------------------------
    # 5. 综合对比仪表盘
    # -------------------------------------------------------------------------

    def plot_comprehensive_dashboard(
        self,
        predictions_dict: Optional[Dict[str, np.ndarray]] = None,
        targets: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (16, 12),
        save: bool = True,
    ) -> plt.Figure:
        """
        绘制综合对比仪表盘

        包含多个子图：
        - 训练/验证Loss
        - 梯度范数
        - 性能指标柱状图
        - 预测结果（如有）
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

        # (1) 训练Loss
        ax1 = fig.add_subplot(gs[0, 0])
        for model_type in self.model_types:
            record = self.records[model_type]
            color = self.MODEL_COLORS.get(model_type, '#000000')
            ax1.plot(record.epochs, record.train_losses, label=model_type,
                    color=color, linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('训练Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # (2) 验证Loss
        ax2 = fig.add_subplot(gs[0, 1])
        for model_type in self.model_types:
            record = self.records[model_type]
            color = self.MODEL_COLORS.get(model_type, '#000000')
            ax2.plot(record.epochs, record.val_losses, label=model_type,
                    color=color, linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('验证Loss', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # (3) 梯度范数
        ax3 = fig.add_subplot(gs[1, 0])
        for model_type in self.model_types:
            record = self.records[model_type]
            total_norms = []
            for grad_dict in record.grad_norms_per_epoch:
                if grad_dict:
                    avg_norm = np.mean(list(grad_dict.values()))
                    total_norms.append(avg_norm)
                else:
                    total_norms.append(0)
            color = self.MODEL_COLORS.get(model_type, '#000000')
            ax3.plot(record.epochs, total_norms, label=model_type,
                    color=color, linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('梯度范数', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        # (4) 性能指标柱状图
        ax4 = fig.add_subplot(gs[1, 1])
        metrics = ['MSE', 'MAE', 'RMSE']
        x = np.arange(len(self.model_types))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = []
            for model_type in self.model_types:
                record = self.records[model_type]
                if metric == 'MSE':
                    values.append(record.final_mse)
                elif metric == 'MAE':
                    values.append(record.final_mae)
                else:
                    values.append(record.final_rmse)
            ax4.bar(x + i * width, values, width, label=metric, alpha=0.8)

        ax4.set_xlabel('模型')
        ax4.set_ylabel('误差值')
        ax4.set_title('性能指标对比', fontweight='bold')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(self.model_types)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # (5) 训练时间对比
        ax5 = fig.add_subplot(gs[2, 0])
        train_times = [self.records[mt].total_train_time for mt in self.model_types]
        colors = [self.MODEL_COLORS.get(mt, '#000000') for mt in self.model_types]
        bars = ax5.bar(self.model_types, train_times, color=colors, alpha=0.8)
        ax5.set_xlabel('模型')
        ax5.set_ylabel('训练时间 (秒)')
        ax5.set_title('训练时间对比', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        # 在柱状图上标注数值
        for bar, time_val in zip(bars, train_times):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.1f}s',
                    ha='center', va='bottom', fontsize=10)

        # (6) 收敛速度对比
        ax6 = fig.add_subplot(gs[2, 1])
        conv_epochs = []
        for model_type in self.model_types:
            record = self.records[model_type]
            conv_epochs.append(record.convergence_epoch if record.convergence_epoch != -1 else record.best_epoch)

        colors = [self.MODEL_COLORS.get(mt, '#000000') for mt in self.model_types]
        bars = ax6.bar(self.model_types, conv_epochs, color=colors, alpha=0.8)
        ax6.set_xlabel('模型')
        ax6.set_ylabel('收敛Epoch')
        ax6.set_title('收敛速度对比（越小越好）', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        for bar, epoch in zip(bars, conv_epochs):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{epoch}',
                    ha='center', va='bottom', fontsize=10)

        fig.suptitle(
            f'{self.dataset_name} - RNN/GRU/LSTM 综合对比仪表盘',
            fontsize=18, fontweight='bold', y=0.98
        )

        if save:
            save_path = os.path.join(
                self.output_dir,
                f'{self.dataset_name}_dashboard.png'
            )
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[可视化] 综合仪表盘已保存: {save_path}")

        return fig

    # -------------------------------------------------------------------------
    # 6. 长期依赖能力可视化
    # -------------------------------------------------------------------------

    @staticmethod
    def plot_long_term_dependency_results(
        results: Dict[int, Dict[str, Dict[str, float]]],
        dataset_name: str,
        output_dir: str = "./figures",
        metric: str = 'mse',
        figsize: Tuple[int, int] = (12, 6),
        save: bool = True,
    ) -> plt.Figure:
        """
        绘制长期依赖能力测试结果

        Args:
            results: 长期依赖测试结果 {seq_length: {model_type: {metric: value}}}
            dataset_name: 数据集名称
            output_dir: 输出目录
            metric: 要绘制的指标 ('mse', 'mae', 'rmse')
            figsize: 图像尺寸
            save: 是否保存
        """
        fig, ax = plt.subplots(figsize=figsize)

        seq_lengths = sorted(results.keys())
        model_types = sorted(next(iter(results.values())).keys())
        colors = {'RNN': '#E74C3C', 'GRU': '#3498DB', 'LSTM': '#2ECC71'}

        for model_type in model_types:
            values = [results[sl][model_type][metric] for sl in seq_lengths]
            color = colors.get(model_type, '#000000')
            ax.plot(seq_lengths, values, marker='o', label=model_type,
                   color=color, linewidth=2.5, markersize=8)

        ax.set_xlabel('序列长度', fontsize=13)
        ax.set_ylabel(f'{metric.upper()}', fontsize=13)
        ax.set_title(
            f'{dataset_name} - 长期依赖能力测试（{metric.upper()}）\n'
            f'序列长度增加时性能变化',
            fontsize=14, fontweight='bold'
        )
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # 添加说明
        ax.text(
            0.02, 0.98,
            '说明：随着序列长度增加，\n'
            'RNN性能下降更明显，\n'
            'LSTM/GRU保持相对稳定',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )

        plt.tight_layout()

        if save:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(
                output_dir,
                f'{dataset_name}_long_term_dependency_{metric}.png'
            )
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[可视化] 长期依赖能力图已保存: {save_path}")

        return fig

    # -------------------------------------------------------------------------
    # 7. 生成所有可视化
    # -------------------------------------------------------------------------

    def generate_all_visualizations(
        self,
        predictions_dict: Optional[Dict[str, np.ndarray]] = None,
        targets: Optional[np.ndarray] = None,
    ):
        """
        生成所有可视化图表

        Args:
            predictions_dict: 预测结果字典
            targets: 真实值
        """
        print("\n" + "=" * 60)
        print("开始生成可视化图表...")
        print("=" * 60)

        # 1. 训练收敛曲线
        self.plot_training_curves(save=True)

        # 2. 梯度范数分析
        self.plot_gradient_norms(save=True)

        # 3. 预测结果对比
        if predictions_dict is not None and targets is not None:
            self.plot_predictions(predictions_dict, targets, save=True)

        # 4. 性能对比表格
        self.generate_performance_table(save=True)

        # 5. 综合仪表盘
        self.plot_comprehensive_dashboard(predictions_dict, targets, save=True)

        print("\n" + "=" * 60)
        print("所有可视化图表生成完成！")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)


# =============================================================================
# 从文件加载并可视化
# =============================================================================

def load_and_visualize(
    dataset_name: str,
    results_dir: str = "./results",
    output_dir: str = "./figures",
    checkpoint_dir: str = "./checkpoints",
) -> ComparisonVisualizer:
    """
    从保存的训练记录文件加载并生成可视化

    Args:
        dataset_name: 数据集名称
        results_dir: 训练记录目录
        output_dir: 图表输出目录
        checkpoint_dir: 模型检查点目录

    Returns:
        ComparisonVisualizer实例
    """
    records = {}
    for model_type in ["RNN", "GRU", "LSTM"]:
        record_path = os.path.join(
            results_dir,
            f"{model_type}_{dataset_name}_record.pkl"
        )
        if os.path.exists(record_path):
            records[model_type] = TrainingRecord.load(record_path)
            print(f"[可视化] 已加载 {model_type} 训练记录")
        else:
            print(f"[可视化] 警告: 未找到 {model_type} 训练记录")

    if not records:
        raise FileNotFoundError(f"未找到任何训练记录文件")

    visualizer = ComparisonVisualizer(
        records=records,
        dataset_name=dataset_name,
        output_dir=output_dir,
    )

    return visualizer


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RNN/GRU/LSTM 可视化分析模块")
    print("=" * 60)

    # 尝试加载已有的训练记录并生成可视化
    dataset_name = "beijing_pm25"

    try:
        visualizer = load_and_visualize(dataset_name)
        visualizer.generate_all_visualizations()
    except FileNotFoundError as e:
        print(f"未找到训练记录: {e}")
        print("请先运行 train_models.py 进行训练")
