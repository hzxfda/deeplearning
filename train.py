# -*- coding: utf-8 -*-
"""
train.py - 训练脚本
一体化的RNN/GRU/LSTM模型训练系统
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List, Optional
import pickle
import time


# ============================================================================
# 数据集
# ============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path: str, seq_length: int = 24, pred_length: int = 6, 
                 split: str = "train", target_col: str = None):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.split = split
        
        # 加载数据
        df = pd.read_csv(data_path)
        
        # 处理缺失值
        df = df.fillna(df.mean(numeric_only=True))
        
        # 确定目标列
        if target_col is None:
            target_col = df.columns[-1] if 'pm2.5' not in df.columns else 'pm2.5'
        
        # 特征处理
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col not in numeric_cols:
            numeric_cols = [target_col] + numeric_cols
        
        self.data = df[numeric_cols].values.astype(np.float32)
        
        # 归一化
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.data)
        
        # 数据划分
        n = len(self.data)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        if split == 'train':
            self.data = self.data[:train_end]
        elif split == 'val':
            self.data = self.data[train_end:val_end]
        else:  # test
            self.data = self.data[val_end:]
        
        # 创建序列
        self.sequences = []
        total_len = seq_length + pred_length
        for i in range(len(self.data) - total_len + 1):
            x = self.data[i:i+seq_length]
            y = self.data[i+seq_length:i+seq_length+pred_length, 0:1]
            self.sequences.append((torch.from_numpy(x), torch.from_numpy(y)))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


# ============================================================================
# 模型
# ============================================================================

class RNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 pred_length: int = 6, model_type: str = "LSTM", dropout: float = 0.2):
        super().__init__()
        self.model_type = model_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        
        # RNN层
        if self.model_type == "RNN":
            rnn_class = nn.RNN
        elif self.model_type == "GRU":
            rnn_class = nn.GRU
        elif self.model_type == "LSTM":
            rnn_class = nn.LSTM
        else:
            raise ValueError(f"不支持的模型: {model_type}")
        
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, x):
        # x: [batch_size, seq_length, input_size]
        out, _ = self.rnn(x)
        # 用最后时刻输出预测所有未来时刻
        last_out = out[:, -1, :]  # [batch_size, hidden_size]
        pred = self.fc(last_out)  # [batch_size, 1]
        # 重复pred_length次
        pred = pred.repeat(1, self.pred_length).reshape(-1, self.pred_length, 1)
        return pred


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, lr: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.criterion = nn.MSELoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'val_rmse': [],
        }
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_mse = 0
        total_mae = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                mse = nn.functional.mse_loss(pred, y)
                mae = nn.functional.l1_loss(pred, y)
                total_mse += mse.item()
                total_mae += mae.item()
        
        avg_mse = total_mse / len(val_loader)
        avg_mae = total_mae / len(val_loader)
        avg_rmse = np.sqrt(avg_mse)
        return avg_mse, avg_mae, avg_rmse
    
    def fit(self, train_loader, val_loader, epochs: int = 100, verbose: bool = True):
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_mse, val_mae, val_rmse = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_mse)
            self.history['val_mse'].append(val_mse)
            self.history['val_mae'].append(val_mae)
            self.history['val_rmse'].append(val_rmse)
            
            self.scheduler.step(val_mse)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_mse={val_mse:.6f}, val_mae={val_mae:.6f}, val_rmse={val_rmse:.6f}")
            
            if val_mse < best_loss:
                best_loss = val_mse
                patience = 0
            else:
                patience += 1
                if patience > 15:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.history


# ============================================================================
# 主训练函数
# ============================================================================

def train_all_models(dataset_path: str, dataset_name: str, epochs: int = 100, 
                     batch_size: int = 64, hidden_size: int = 64, num_layers: int = 2):
    """训练三种模型"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    print(f"数据集: {dataset_name}")
    print("=" * 70)
    
    # 创建数据加载器
    train_dataset = TimeSeriesDataset(dataset_path, split='train')
    val_dataset = TimeSeriesDataset(dataset_path, split='val')
    test_dataset = TimeSeriesDataset(dataset_path, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    input_size = train_dataset.data.shape[1]
    
    results = {}
    
    # 训练三种模型
    for model_type in ['RNN', 'GRU', 'LSTM']:
        print(f"\n训练 {model_type} 模型...")
        print("-" * 70)
        
        model = RNNModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            model_type=model_type,
        )
        
        trainer = Trainer(model, device)
        history = trainer.fit(train_loader, val_loader, epochs=epochs)
        
        # 测试集评估
        model.eval()
        test_mse, test_mae, test_rmse = trainer.validate(test_loader)
        
        results[model_type] = {
            'history': history,
            'test_mse': float(test_mse),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
        }
        
        # 保存模型
        os.makedirs('./checkpoints', exist_ok=True)
        save_path = f"./checkpoints/{model_type}_{dataset_name}_best.pt"
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存: {save_path}")
        print(f"测试集 - MSE: {test_mse:.6f}, MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")
    
    # 保存结果
    os.makedirs('./results', exist_ok=True)
    result_path = f"./results/summary_{dataset_name}.json"
    with open(result_path, 'w') as f:
        # 将numpy数值转换为Python基础类型
        json_results = {}
        for model_type, data in results.items():
            json_results[model_type] = {
                'test_mse': float(data['test_mse']),
                'test_mae': float(data['test_mae']),
                'test_rmse': float(data['test_rmse']),
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n结果已保存: {result_path}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练RNN/GRU/LSTM模型')
    parser.add_argument('--dataset', type=str, default='beijing_pm25',
                        choices=['beijing_pm25', 'california_traffic'],
                        help='数据集名称')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--hidden_size', type=int, default=64, help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, help='RNN层数')
    
    args = parser.parse_args()
    
    dataset_path = f"./data/{args.dataset}.csv"
    if os.path.exists(dataset_path):
        train_all_models(
            dataset_path=dataset_path,
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )
    else:
        print(f"数据集文件不存在: {dataset_path}")
