# -*- coding: utf-8 -*-
"""
train_models.py
RNN/GRU/LSTM 三模型统一训练框架
用于智慧城市时序预测任务的对比训练
"""

import os
import time
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

from datasets import get_dataloaders


# =============================================================================
# 模型定义
# =============================================================================

class SleepRNN(nn.Module):
    """
    统一的RNN/GRU/LSTM模型类
    支持三种循环神经网络结构
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        pred_length: int = 6,
        model_type: str = "LSTM",
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """
        初始化循环神经网络模型

        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: RNN层数
            output_size: 输出维度
            pred_length: 预测步长
            model_type: 模型类型 ('RNN', 'GRU', 'LSTM')
            dropout: Dropout比率
            bidirectional: 是否使用双向结构
        """
        super().__init__()
        self.model_type = model_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.pred_length = pred_length

        # 选择RNN单元类型
        if self.model_type == "RNN":
            rnn_class = nn.RNN
        elif self.model_type == "GRU":
            rnn_class = nn.GRU
        elif self.model_type == "LSTM":
            rnn_class = nn.LSTM
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # RNN层
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 输出层：将RNN输出映射到预测序列
        # 使用所有时间步的输出进行预测
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

        # 序列到序列的预测：使用全连接层将seq_length映射到pred_length
        self.seq_proj = nn.Linear(pred_length, pred_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_length, input_size]

        Returns:
            预测序列 [batch_size, pred_length, output_size]
        """
        # RNN前向传播
        # out: [batch_size, seq_length, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hidden_size]
        out, hidden = self.rnn(x)

        # 使用所有时间步的输出进行预测
        # [batch_size, seq_length, output_size]
        predictions = self.fc(out)

        # 取最后pred_length个时间步的预测结果
        # 如果seq_length >= pred_length，取最后pred_length步
        if predictions.size(1) >= self.pred_length:
            output = predictions[:, -self.pred_length:, :]
        else:
            # 如果序列长度不足，重复最后一个值
            last = predictions[:, -1:, :].repeat(1, self.pred_length - predictions.size(1), 1)
            output = torch.cat([predictions, last], dim=1)

        return output

    def get_layer_gradients(self) -> Dict[str, float]:
        """
        获取每一层的梯度范数，用于分析梯度消散

        Returns:
            层名到梯度范数的字典
        """
        grad_norms = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
        return grad_norms


# =============================================================================
# 训练记录器
# =============================================================================

@dataclass
class TrainingRecord:
    """训练记录数据类"""
    model_type: str
    dataset_name: str
    epochs: List[int] = None
    train_losses: List[float] = None
    val_losses: List[float] = None
    grad_norms_per_epoch: List[Dict[str, float]] = None
    learning_rates: List[float] = None
    epoch_times: List[float] = None
    best_val_loss: float = float('inf')
    best_epoch: int = 0
    total_train_time: float = 0.0
    final_mse: float = 0.0
    final_mae: float = 0.0
    final_rmse: float = 0.0
    convergence_epoch: int = -1  # 达到特定loss阈值所需的epoch
    convergence_threshold: float = 0.01

    def __post_init__(self):
        if self.epochs is None:
            self.epochs = []
        if self.train_losses is None:
            self.train_losses = []
        if self.val_losses is None:
            self.val_losses = []
        if self.grad_norms_per_epoch is None:
            self.grad_norms_per_epoch = []
        if self.learning_rates is None:
            self.learning_rates = []
        if self.epoch_times is None:
            self.epoch_times = []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def save(self, path: str):
        """保存训练记录"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'TrainingRecord':
        """加载训练记录"""
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# 训练器
# =============================================================================

class ModelTrainer:
    """
    模型训练器
    支持RNN/GRU/LSTM三种模型的训练和对比
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """
        初始化训练器

        Args:
            model: 神经网络模型
            model_type: 模型类型名称
            device: 计算设备
            checkpoint_dir: 检查点保存目录
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.model = model.to(device)
        self.model_type = model_type
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 优化器和学习率调度器
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 训练记录
        self.record = TrainingRecord(model_type=model_type, dataset_name="")

        # 断点续训状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        训练一个epoch

        Returns:
            (平均训练loss, 平均梯度范数字典)
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_grad_norms: Dict[str, List[float]] = {}

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)

            # 反向传播
            loss.backward()

            # 记录梯度范数
            layer_grads = self.model.get_layer_gradients()
            for name, grad_norm in layer_grads.items():
                if name not in epoch_grad_norms:
                    epoch_grad_norms[name] = []
                epoch_grad_norms[name].append(grad_norm)

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples

        # 计算平均梯度范数
        avg_grad_norms = {
            name: np.mean(values)
            for name, values in epoch_grad_norms.items()
        }

        return avg_loss, avg_grad_norms

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float, float]:
        """
        验证模型

        Returns:
            (MSE, MAE, RMSE)
        """
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)

                mse = nn.functional.mse_loss(output, y, reduction='sum')
                mae = nn.functional.l1_loss(output, y, reduction='sum')

                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += y.numel()

        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        avg_rmse = np.sqrt(avg_mse)

        return avg_mse, avg_mae, avg_rmse

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        dataset_name: str,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        convergence_threshold: float = 0.01,
        verbose: bool = True,
    ) -> TrainingRecord:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            dataset_name: 数据集名称
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            convergence_threshold: 收敛阈值（loss低于此值认为收敛）
            verbose: 是否打印训练信息

        Returns:
            训练记录
        """
        self.record.dataset_name = dataset_name
        self.record.convergence_threshold = convergence_threshold

        patience_counter = 0
        total_train_start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"开始训练 {self.model_type} 模型")
            print(f"数据集: {dataset_name}")
            print(f"总epoch数: {num_epochs}")
            print(f"{'='*60}")

        for epoch in range(self.current_epoch, num_epochs):
            epoch_start = time.time()

            # 训练
            train_loss, grad_norms = self.train_epoch(train_loader)

            # 验证
            val_mse, val_mae, val_rmse = self.validate(val_loader)
            val_loss = val_mse

            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            epoch_time = time.time() - epoch_start

            # 记录
            self.record.epochs.append(epoch + 1)
            self.record.train_losses.append(train_loss)
            self.record.val_losses.append(val_loss)
            self.record.grad_norms_per_epoch.append(grad_norms)
            self.record.learning_rates.append(current_lr)
            self.record.epoch_times.append(epoch_time)

            # 检查收敛
            if self.record.convergence_epoch == -1 and val_loss < convergence_threshold:
                self.record.convergence_epoch = epoch + 1

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.record.best_val_loss = val_loss
                self.record.best_epoch = epoch + 1
                self.save_checkpoint(epoch + 1, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1

            # 打印进度
            if verbose and (epoch + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch+1:03d}/{num_epochs}] "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val MSE: {val_mse:.6f} | "
                    f"Val MAE: {val_mae:.6f} | "
                    f"Val RMSE: {val_rmse:.6f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.2f}s"
                )

            # 早停
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\n早停触发！连续 {early_stopping_patience} 个epoch验证loss未改善")
                break

        total_train_time = time.time() - total_train_start
        self.record.total_train_time = total_train_time
        self.record.final_mse = val_mse
        self.record.final_mae = val_mae
        self.record.final_rmse = val_rmse

        if verbose:
            print(f"\n训练完成！总用时: {total_train_time:.2f}s")
            print(f"最佳验证Loss: {self.record.best_val_loss:.6f} (Epoch {self.record.best_epoch})")
            if self.record.convergence_epoch != -1:
                print(f"收敛epoch: {self.record.convergence_epoch}")
            else:
                print(f"未达到收敛阈值 ({convergence_threshold})")

        return self.record

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_type': self.model_type,
        }

        # 保存最新检查点
        latest_path = os.path.join(
            self.checkpoint_dir,
            f"{self.model_type}_{self.record.dataset_name}_latest.pt"
        )
        torch.save(checkpoint, latest_path)

        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir,
                f"{self.model_type}_{self.record.dataset_name}_best.pt"
            )
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        加载模型检查点，支持断点续训

        Returns:
            加载的epoch数
        """
        if not os.path.exists(checkpoint_path):
            print(f"检查点不存在: {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        loaded_epoch = checkpoint['epoch']
        self.current_epoch = loaded_epoch

        print(f"已加载检查点: {checkpoint_path} (Epoch {loaded_epoch})")
        return loaded_epoch

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        在测试集上评估模型

        Returns:
            (MSE, MAE, RMSE, 预测值数组, 真实值数组)
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)

                mse = nn.functional.mse_loss(output, y, reduction='sum')
                mae = nn.functional.l1_loss(output, y, reduction='sum')

                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += y.numel()

                all_preds.append(output.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        avg_rmse = np.sqrt(avg_mse)

        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        return avg_mse, avg_mae, avg_rmse, predictions, targets


# =============================================================================
# 三模型对比训练系统
# =============================================================================

class MultiModelComparison:
    """
    RNN/GRU/LSTM 三模型对比训练系统
    同时训练三种模型并记录对比指标
    """

    MODEL_TYPES = ["RNN", "GRU", "LSTM"]

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./data",
        seq_length: int = 24,
        pred_length: int = 6,
        batch_size: int = 64,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        checkpoint_dir: str = "./checkpoints",
        results_dir: str = "./results",
        device: Optional[torch.device] = None,
    ):
        """
        初始化多模型对比系统

        Args:
            dataset_name: 数据集名称
            data_dir: 数据目录
            seq_length: 输入序列长度
            pred_length: 预测序列长度
            batch_size: 批量大小
            hidden_size: 隐藏层维度
            num_layers: RNN层数
            num_epochs: 训练轮数
            learning_rate: 学习率
            checkpoint_dir: 检查点目录
            results_dir: 结果保存目录
            device: 计算设备
        """
        self.dataset_name = dataset_name
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # 设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        print(f"使用设备: {self.device}")

        # 加载数据
        self.train_loader, self.val_loader, self.test_loader, self.dataset_info = \
            get_dataloaders(
                dataset_name=dataset_name,
                data_dir=data_dir,
                seq_length=seq_length,
                pred_length=pred_length,
                batch_size=batch_size,
            )

        self.input_size = self.dataset_info['input_size']
        self.output_size = self.dataset_info['output_size']

        # 训练结果
        self.trainers: Dict[str, ModelTrainer] = {}
        self.records: Dict[str, TrainingRecord] = {}

    def build_model(self, model_type: str) -> SleepRNN:
        """构建指定类型的模型"""
        return SleepRNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            pred_length=self.pred_length,
            model_type=model_type,
            dropout=0.2,
            bidirectional=False,
        )

    def train_all_models(
        self,
        resume: bool = False,
        convergence_threshold: float = 0.01,
    ) -> Dict[str, TrainingRecord]:
        """
        训练所有三种模型

        Args:
            resume: 是否从检查点恢复训练
            convergence_threshold: 收敛阈值

        Returns:
            各模型的训练记录字典
        """
        print(f"\n{'#'*60}")
        print(f"# 开始三模型对比训练")
        print(f"# 数据集: {self.dataset_info['task_name']}")
        print(f"# 输入维度: {self.input_size}, 输出维度: {self.output_size}")
        print(f"# 序列长度: {self.seq_length}, 预测长度: {self.pred_length}")
        print(f"# 隐藏层: {self.hidden_size}, 层数: {self.num_layers}")
        print(f"{'#'*60}")

        for model_type in self.MODEL_TYPES:
            print(f"\n{'-'*60}")
            print(f"训练 {model_type} 模型")
            print(f"{'-'*60}")

            # 构建模型和训练器
            model = self.build_model(model_type)
            trainer = ModelTrainer(
                model=model,
                model_type=model_type,
                device=self.device,
                checkpoint_dir=self.checkpoint_dir,
                learning_rate=self.learning_rate,
            )

            # 断点续训
            if resume:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"{model_type}_{self.dataset_name}_latest.pt"
                )
                trainer.load_checkpoint(checkpoint_path)

            # 训练
            record = trainer.fit(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                dataset_name=self.dataset_name,
                num_epochs=self.num_epochs,
                convergence_threshold=convergence_threshold,
            )

            # 测试集评估
            test_mse, test_mae, test_rmse, preds, targets = trainer.evaluate(self.test_loader)
            record.final_mse = test_mse
            record.final_mae = test_mae
            record.final_rmse = test_rmse

            print(f"\n{model_type} 测试集结果:")
            print(f"  MSE:  {test_mse:.6f}")
            print(f"  MAE:  {test_mae:.6f}")
            print(f"  RMSE: {test_rmse:.6f}")

            # 保存结果
            self.trainers[model_type] = trainer
            self.records[model_type] = record

            # 保存训练记录
            record_path = os.path.join(
                self.results_dir,
                f"{model_type}_{self.dataset_name}_record.pkl"
            )
            record.save(record_path)

        # 保存汇总结果
        self._save_summary()

        return self.records

    def _save_summary(self):
        """保存训练汇总结果"""
        summary = {}
        for model_type, record in self.records.items():
            summary[model_type] = {
                'best_val_loss': record.best_val_loss,
                'best_epoch': record.best_epoch,
                'final_mse': record.final_mse,
                'final_mae': record.final_mae,
                'final_rmse': record.final_rmse,
                'total_train_time': record.total_train_time,
                'convergence_epoch': record.convergence_epoch,
            }

        summary_path = os.path.join(
            self.results_dir,
            f"summary_{self.dataset_name}.json"
        )
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n汇总结果已保存: {summary_path}")

    def load_records(self) -> Dict[str, TrainingRecord]:
        """从文件加载训练记录"""
        self.records = {}
        for model_type in self.MODEL_TYPES:
            record_path = os.path.join(
                self.results_dir,
                f"{model_type}_{self.dataset_name}_record.pkl"
            )
            if os.path.exists(record_path):
                self.records[model_type] = TrainingRecord.load(record_path)
                print(f"已加载 {model_type} 训练记录")
        return self.records

    def get_test_predictions(self, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定模型在测试集上的预测结果

        Returns:
            (预测值, 真实值)
        """
        if model_type not in self.trainers:
            # 尝试加载模型
            model = self.build_model(model_type)
            trainer = ModelTrainer(
                model=model,
                model_type=model_type,
                device=self.device,
                checkpoint_dir=self.checkpoint_dir,
            )
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{model_type}_{self.dataset_name}_best.pt"
            )
            trainer.load_checkpoint(checkpoint_path)
            self.trainers[model_type] = trainer

        _, _, _, preds, targets = self.trainers[model_type].evaluate(self.test_loader)
        return preds, targets


# =============================================================================
# 长期依赖能力测试
# =============================================================================

class LongTermDependencyTest:
    """
    长期依赖能力测试
    测试不同序列长度下模型的预测性能
    """

    def __init__(
        self,
        dataset_name: str,
        seq_lengths: List[int] = None,
        data_dir: str = "./data",
        pred_length: int = 6,
        num_epochs: int = 50,
        device: Optional[torch.device] = None,
    ):
        """
        初始化长期依赖测试

        Args:
            dataset_name: 数据集名称
            seq_lengths: 要测试的序列长度列表
            data_dir: 数据目录
            pred_length: 预测长度
            num_epochs: 每个序列长度的训练轮数
            device: 计算设备
        """
        self.dataset_name = dataset_name
        self.seq_lengths = seq_lengths or [6, 12, 24, 48, 72, 96]
        self.data_dir = data_dir
        self.pred_length = pred_length
        self.num_epochs = num_epochs

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.results: Dict[int, Dict[str, Dict[str, float]]] = {}

    def run(self, model_types: List[str] = None) -> Dict:
        """
        运行长期依赖能力测试

        Returns:
            测试结果字典
        """
        model_types = model_types or ["RNN", "GRU", "LSTM"]

        print(f"\n{'#'*60}")
        print(f"# 长期依赖能力测试")
        print(f"# 数据集: {self.dataset_name}")
        print(f"# 测试序列长度: {self.seq_lengths}")
        print(f"{'#'*60}")

        for seq_len in self.seq_lengths:
            print(f"\n{'='*60}")
            print(f"序列长度: {seq_len}")
            print(f"{'='*60}")

            self.results[seq_len] = {}

            # 加载数据
            train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
                dataset_name=self.dataset_name,
                data_dir=self.data_dir,
                seq_length=seq_len,
                pred_length=self.pred_length,
                batch_size=64,
            )

            for model_type in model_types:
                print(f"\n  测试 {model_type}...")

                model = SleepRNN(
                    input_size=dataset_info['input_size'],
                    hidden_size=64,
                    num_layers=2,
                    output_size=1,
                    pred_length=self.pred_length,
                    model_type=model_type,
                )

                trainer = ModelTrainer(
                    model=model,
                    model_type=model_type,
                    device=self.device,
                    learning_rate=1e-3,
                )

                # 训练
                trainer.fit(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    dataset_name=self.dataset_name,
                    num_epochs=self.num_epochs,
                    verbose=False,
                )

                # 测试
                test_mse, test_mae, test_rmse, _, _ = trainer.evaluate(test_loader)

                self.results[seq_len][model_type] = {
                    'mse': test_mse,
                    'mae': test_mae,
                    'rmse': test_rmse,
                }

                print(f"    MSE: {test_mse:.6f}, MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")

        return self.results


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    # 示例：训练北京PM2.5数据集
    print("=" * 60)
    print("RNN/GRU/LSTM 三模型对比训练系统")
    print("=" * 60)

    # 检查CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建对比系统并训练
    comparison = MultiModelComparison(
        dataset_name="beijing_pm25",
        seq_length=24,
        pred_length=6,
        batch_size=64,
        hidden_size=64,
        num_layers=2,
        num_epochs=30,  # 示例用较少的epoch
        device=device,
    )

    records = comparison.train_all_models()

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    for model_type, record in records.items():
        print(f"\n{model_type}:")
        print(f"  最佳Val Loss: {record.best_val_loss:.6f}")
        print(f"  测试 MSE: {record.final_mse:.6f}")
        print(f"  测试 MAE: {record.final_mae:.6f}")
        print(f"  测试 RMSE: {record.final_rmse:.6f}")
        print(f"  训练时间: {record.total_train_time:.2f}s")
