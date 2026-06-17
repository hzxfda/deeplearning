# -*- coding: utf-8 -*-
"""
datasets.py
智慧城市时序预测数据集加载器
支持北京PM2.5数据集和加州交通流量数据集
"""

import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional, List


# =============================================================================
# 数据集下载工具
# =============================================================================

class DatasetDownloader:
    """数据集下载器，负责从网上下载公开数据集"""

    # 北京PM2.5数据集来源：UCI Machine Learning Repository
    BEIJING_PM25_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00381/PRSA_data_2010.1.1-2014.12.31.csv"
    )

    # 加州交通数据集来源：Caltrans PeMS
    # 使用一个公开的交通流量数据集（PeMSD7）
    CALIFORNIA_TRAFFIC_URL = (
        "https://raw.githubusercontent.com/Davidham3/STSGCN/master/data/"
        "PEMS04/PEMS04.csv"
    )

    def __init__(self, data_dir: str = "./data"):
        """
        初始化下载器

        Args:
            data_dir: 数据保存目录
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_beijing_pm25(self) -> str:
        """
        下载北京PM2.5数据集

        Returns:
            下载后的本地文件路径
        """
        local_path = os.path.join(self.data_dir, "beijing_pm25.csv")
        if os.path.exists(local_path):
            print(f"[数据集] 北京PM2.5数据已存在: {local_path}")
            return local_path

        print(f"[数据集] 正在下载北京PM2.5数据集...")
        print(f"[数据集] 来源: UCI Machine Learning Repository")
        try:
            urllib.request.urlretrieve(self.BEIJING_PM25_URL, local_path)
            print(f"[数据集] 下载完成: {local_path}")
        except Exception as e:
            print(f"[数据集] 下载失败: {e}")
            print("[数据集] 尝试使用备用数据生成...")
            self._generate_beijing_pm25_fallback(local_path)
        return local_path

    def download_california_traffic(self) -> str:
        """
        下载加州交通流量数据集

        Returns:
            下载后的本地文件路径
        """
        local_path = os.path.join(self.data_dir, "california_traffic.csv")
        if os.path.exists(local_path):
            print(f"[数据集] 加州交通数据已存在: {local_path}")
            return local_path

        print(f"[数据集] 正在下载加州交通数据集...")
        try:
            urllib.request.urlretrieve(self.CALIFORNIA_TRAFFIC_URL, local_path)
            print(f"[数据集] 下载完成: {local_path}")
        except Exception as e:
            print(f"[数据集] 下载失败: {e}")
            print("[数据集] 尝试使用备用数据生成...")
            self._generate_california_traffic_fallback(local_path)
        return local_path

    def _generate_beijing_pm25_fallback(self, save_path: str):
        """
        生成北京PM2.5备用数据（当下载失败时使用）
        模拟真实的气象和PM2.5数据特征
        """
        print("[数据集] 生成北京PM2.5备用数据...")
        np.random.seed(42)
        n_samples = 43824  # 5年小时级数据

        # 生成时间序列
        dates = pd.date_range(start='2010-01-01', periods=n_samples, freq='H')

        # 模拟季节性温度变化
        hour_of_day = np.arange(n_samples) % 24
        day_of_year = np.arange(n_samples) % 8760 / 8760 * 2 * np.pi

        # 温度：季节性 + 日变化
        temperature = (
            15 + 15 * np.sin(day_of_year - np.pi/2) +  # 季节性
            5 * np.sin(hour_of_day / 24 * 2 * np.pi) +   # 日变化
            np.random.normal(0, 2, n_samples)             # 噪声
        )

        # 风速：与温度负相关，冬季风大
        wind_speed = (
            3 + 2 * np.sin(day_of_year + np.pi) +
            np.random.exponential(1, n_samples)
        )
        wind_speed = np.clip(wind_speed, 0, 15)

        # 降雨：随机事件，夏季较多
        rain_prob = 0.05 + 0.1 * np.maximum(0, np.sin(day_of_year))
        rainfall = np.random.binomial(1, rain_prob) * np.random.exponential(2, n_samples)

        # 气压：季节性变化
        pressure = (
            1013 + 15 * np.sin(day_of_year) +
            np.random.normal(0, 5, n_samples)
        )

        # 湿度：与降雨和温度相关
        humidity = (
            50 + 20 * np.sin(day_of_year + np.pi/3) -
            0.5 * temperature +
            10 * (rainfall > 0) +
            np.random.normal(0, 5, n_samples)
        )
        humidity = np.clip(humidity, 10, 100)

        # PM2.5：与气象条件复杂相关
        # 冬季采暖 + 静稳天气导致高浓度
        base_pm25 = (
            80 + 60 * np.sin(day_of_year + np.pi/2) +  # 冬季高
            -2 * wind_speed +                            # 风速稀释
            -1 * rainfall +                              # 降雨清除
            0.5 * humidity +                             # 湿度影响
            np.random.normal(0, 20, n_samples)           # 噪声
        )
        # 沙尘暴事件（随机高值）
        dust_storms = np.random.binomial(1, 0.001, n_samples) * np.random.uniform(200, 500, n_samples)
        pm25 = np.maximum(5, base_pm25 + dust_storms)

        df = pd.DataFrame({
            'No': range(1, n_samples + 1),
            'year': dates.year,
            'month': dates.month,
            'day': dates.day,
            'hour': dates.hour,
            'pm2.5': pm25,
            'DEWP': temperature - 5 + np.random.normal(0, 2, n_samples),  # 露点
            'TEMP': temperature,
            'PRES': pressure,
            'cbwd': np.random.choice(['NW', 'NE', 'SE', 'cv'], n_samples),
            'Iws': wind_speed,
            'Is': rainfall,
            'Ir': np.random.binomial(1, 0.02, n_samples) * np.random.exponential(1, n_samples),
        })
        df.to_csv(save_path, index=False)
        print(f"[数据集] 备用数据已生成: {save_path}")

    def _generate_california_traffic_fallback(self, save_path: str):
        """
        生成加州交通流量备用数据（当下载失败时使用）
        模拟真实的高速公路交通流量特征，包含工作日/周末双重周期性
        """
        print("[数据集] 生成加州交通备用数据...")
        np.random.seed(42)
        n_samples = 26208  # 3年小时级数据 (365*3*24)
        n_sensors = 10     # 10个监测点

        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='h')

        data_dict = {'timestamp': dates}

        for sensor_id in range(n_sensors):
            # 基础流量
            base_flow = 100 + sensor_id * 20

            # 工作日/周末模式
            is_weekend = np.array([d.weekday() >= 5 for d in dates])
            weekend_factor = np.where(is_weekend, 0.6, 1.0)

            # 日周期性：早高峰(8点)、晚高峰(18点)
            hour = np.arange(n_samples) % 24
            morning_peak = 50 * np.exp(-((hour - 8) ** 2) / 4)
            evening_peak = 60 * np.exp(-((hour - 18) ** 2) / 6)
            night_valley = -30 * np.exp(-((hour - 3) ** 2) / 2)

            # 周周期性
            day_of_week = np.array([d.weekday() for d in dates])
            weekly_pattern = 20 * np.sin(day_of_week / 7 * 2 * np.pi)

            # 长期趋势（缓慢增长）
            trend = np.arange(n_samples) / n_samples * 30

            # 随机事件（事故、天气等）
            events = np.random.binomial(1, 0.005, n_samples) * np.random.uniform(-50, -20, n_samples)

            # 噪声
            noise = np.random.normal(0, 10, n_samples)

            flow = (
                base_flow +
                morning_peak +
                evening_peak +
                night_valley +
                weekly_pattern +
                trend +
                events +
                noise
            ) * weekend_factor

            flow = np.maximum(10, flow)
            data_dict[f'sensor_{sensor_id}'] = flow

        df = pd.DataFrame(data_dict)
        df.to_csv(save_path, index=False)
        print(f"[数据集] 备用数据已生成: {save_path}")


# =============================================================================
# 北京PM2.5数据集
# =============================================================================

class BeijingPM25Dataset(Dataset):
    """
    北京PM2.5数据集
    特征包括风速、降雨、温度、气压、湿度等气象要素
    预测未来PM2.5值（用于沙尘暴预警等智慧城市应用）
    """

    def __init__(
        self,
        data_dir: str = "./data",
        seq_length: int = 24,
        pred_length: int = 6,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        download: bool = True,
    ):
        """
        初始化北京PM2.5数据集

        Args:
            data_dir: 数据目录
            seq_length: 输入序列长度（过去多少小时）
            pred_length: 预测序列长度（未来多少小时）
            split: 数据划分 ('train', 'val', 'test')
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            download: 是否自动下载数据
        """
        super().__init__()
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.split = split

        # 下载数据
        if download:
            downloader = DatasetDownloader(data_dir)
            data_path = downloader.download_beijing_pm25()
        else:
            data_path = os.path.join(data_dir, "beijing_pm25.csv")

        # 加载数据
        self.df = pd.read_csv(data_path)
        self._preprocess()
        self._split_data(train_ratio, val_ratio)
        self._create_sequences()

    def _preprocess(self):
        """数据预处理：处理缺失值、编码分类变量、归一化"""
        # 处理PM2.5缺失值
        self.df['pm2.5'] = self.df['pm2.5'].ffill().bfill()

        # 处理其他数值列缺失值
        numeric_cols = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())

        # 风向编码（独热编码）
        if 'cbwd' in self.df.columns:
            wind_dummies = pd.get_dummies(self.df['cbwd'], prefix='wind')
            self.df = pd.concat([self.df, wind_dummies], axis=1)
            self.df = self.df.drop('cbwd', axis=1)

        # 构建特征矩阵
        feature_cols = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
        # 添加风向独热编码列
        wind_cols = [c for c in self.df.columns if c.startswith('wind_')]
        feature_cols.extend(wind_cols)

        # 时间特征
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        feature_cols.extend(['hour_sin', 'hour_cos', 'month_sin', 'month_cos'])

        self.feature_cols = feature_cols
        self.target_col = 'pm2.5'

        # 提取特征和目标
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.targets = self.df[self.target_col].values.astype(np.float32).reshape(-1, 1)

        # 归一化
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        self.features = self.feature_scaler.fit_transform(self.features)
        self.targets = self.target_scaler.fit_transform(self.targets)

    def _split_data(self, train_ratio: float, val_ratio: float):
        """划分训练/验证/测试集"""
        n = len(self.features)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if self.split == 'train':
            self.features = self.features[:train_end]
            self.targets = self.targets[:train_end]
        elif self.split == 'val':
            self.features = self.features[train_end:val_end]
            self.targets = self.targets[train_end:val_end]
        else:  # test
            self.features = self.features[val_end:]
            self.targets = self.targets[val_end:]

    def _create_sequences(self):
        """创建时间序列样本"""
        self.samples = []
        total_len = self.seq_length + self.pred_length
        for i in range(len(self.features) - total_len + 1):
            x = self.features[i : i + self.seq_length]
            y = self.targets[i + self.seq_length : i + total_len]
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """将归一化后的目标值反变换回原始尺度"""
        return self.target_scaler.inverse_transform(y)

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_cols


# =============================================================================
# 加州交通流量数据集
# =============================================================================

class CaliforniaTrafficDataset(Dataset):
    """
    加州交通流量数据集
    捕捉工作日/周末双重周期性
    预测未来交通流量（用于智慧城市交通管理和拥堵预警）
    """

    def __init__(
        self,
        data_dir: str = "./data",
        seq_length: int = 24,
        pred_length: int = 6,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        download: bool = True,
        target_sensor: str = "sensor_0",
    ):
        """
        初始化加州交通流量数据集

        Args:
            data_dir: 数据目录
            seq_length: 输入序列长度（过去多少小时）
            pred_length: 预测序列长度（未来多少小时）
            split: 数据划分 ('train', 'val', 'test')
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            download: 是否自动下载数据
            target_sensor: 目标监测点
        """
        super().__init__()
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.split = split
        self.target_sensor = target_sensor

        # 下载数据
        if download:
            downloader = DatasetDownloader(data_dir)
            data_path = downloader.download_california_traffic()
        else:
            data_path = os.path.join(data_dir, "california_traffic.csv")

        # 加载数据
        self.df = pd.read_csv(data_path)
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        self._preprocess()
        self._split_data(train_ratio, val_ratio)
        self._create_sequences()

    def _preprocess(self):
        """数据预处理：提取时间特征、归一化"""
        # 获取传感器列
        sensor_cols = [c for c in self.df.columns if c.startswith('sensor_')]
        if len(sensor_cols) == 0:
            # 如果是下载的真实数据，可能有不同的列名
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            sensor_cols = [c for c in numeric_cols if c != 'timestamp']

        self.sensor_cols = sensor_cols

        # 时间特征
        if 'timestamp' in self.df.columns:
            timestamps = pd.to_datetime(self.df['timestamp'])
        else:
            timestamps = pd.date_range(start='2020-01-01', periods=len(self.df), freq='h')

        self.df['hour'] = timestamps.dt.hour
        self.df['dayofweek'] = timestamps.dt.dayofweek
        self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(float)

        # 周期性编码
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['dow_sin'] = np.sin(2 * np.pi * self.df['dayofweek'] / 7)
        self.df['dow_cos'] = np.cos(2 * np.pi * self.df['dayofweek'] / 7)

        # 构建特征矩阵
        feature_cols = sensor_cols.copy()
        feature_cols.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend'])

        self.feature_cols = feature_cols
        self.target_col = self.target_sensor if self.target_sensor in self.df.columns else sensor_cols[0]

        # 提取特征和目标
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.targets = self.df[self.target_col].values.astype(np.float32).reshape(-1, 1)

        # 处理缺失值
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)

        # 归一化
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        self.features = self.feature_scaler.fit_transform(self.features)
        self.targets = self.target_scaler.fit_transform(self.targets)

    def _split_data(self, train_ratio: float, val_ratio: float):
        """划分训练/验证/测试集"""
        n = len(self.features)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if self.split == 'train':
            self.features = self.features[:train_end]
            self.targets = self.targets[:train_end]
        elif self.split == 'val':
            self.features = self.features[train_end:val_end]
            self.targets = self.targets[train_end:val_end]
        else:  # test
            self.features = self.features[val_end:]
            self.targets = self.targets[val_end:]

    def _create_sequences(self):
        """创建时间序列样本"""
        self.samples = []
        total_len = self.seq_length + self.pred_length
        for i in range(len(self.features) - total_len + 1):
            x = self.features[i : i + self.seq_length]
            y = self.targets[i + self.seq_length : i + total_len]
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """将归一化后的目标值反变换回原始尺度"""
        return self.target_scaler.inverse_transform(y)

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_cols


# =============================================================================
# 数据加载器工厂函数
# =============================================================================

def get_dataloaders(
    dataset_name: str,
    data_dir: str = "./data",
    seq_length: int = 24,
    pred_length: int = 6,
    batch_size: int = 64,
    num_workers: int = 0,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    获取指定数据集的数据加载器

    Args:
        dataset_name: 数据集名称 ('beijing_pm25' 或 'california_traffic')
        data_dir: 数据目录
        seq_length: 输入序列长度
        pred_length: 预测序列长度
        batch_size: 批量大小
        num_workers: 数据加载线程数
        download: 是否自动下载

    Returns:
        (train_loader, val_loader, test_loader, dataset_info)
    """
    if dataset_name == 'beijing_pm25':
        train_ds = BeijingPM25Dataset(
            data_dir, seq_length, pred_length, 'train', download=download
        )
        val_ds = BeijingPM25Dataset(
            data_dir, seq_length, pred_length, 'val', download=False
        )
        test_ds = BeijingPM25Dataset(
            data_dir, seq_length, pred_length, 'test', download=False
        )
        input_size = len(train_ds.get_feature_names())
        output_size = 1
        task_name = "北京PM2.5预测"

    elif dataset_name == 'california_traffic':
        train_ds = CaliforniaTrafficDataset(
            data_dir, seq_length, pred_length, 'train', download=download
        )
        val_ds = CaliforniaTrafficDataset(
            data_dir, seq_length, pred_length, 'val', download=False
        )
        test_ds = CaliforniaTrafficDataset(
            data_dir, seq_length, pred_length, 'test', download=False
        )
        input_size = len(train_ds.get_feature_names())
        output_size = 1
        task_name = "加州交通流量预测"

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    dataset_info = {
        'name': dataset_name,
        'task_name': task_name,
        'input_size': input_size,
        'output_size': output_size,
        'seq_length': seq_length,
        'pred_length': pred_length,
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'test_size': len(test_ds),
        'feature_names': train_ds.get_feature_names(),
        'train_dataset': train_ds,
        'val_dataset': val_ds,
        'test_dataset': test_ds,
    }

    return train_loader, val_loader, test_loader, dataset_info


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试北京PM2.5数据集")
    print("=" * 60)
    train_loader, val_loader, test_loader, info = get_dataloaders(
        'beijing_pm25', batch_size=32, seq_length=24, pred_length=6
    )
    print(f"任务: {info['task_name']}")
    print(f"输入维度: {info['input_size']}")
    print(f"训练样本数: {info['train_size']}")
    print(f"验证样本数: {info['val_size']}")
    print(f"测试样本数: {info['test_size']}")

    for x, y in train_loader:
        print(f"输入形状: {x.shape}, 目标形状: {y.shape}")
        break

    print("\n" + "=" * 60)
    print("测试加州交通数据集")
    print("=" * 60)
    train_loader, val_loader, test_loader, info = get_dataloaders(
        'california_traffic', batch_size=32, seq_length=24, pred_length=6
    )
    print(f"任务: {info['task_name']}")
    print(f"输入维度: {info['input_size']}")
    print(f"训练样本数: {info['train_size']}")
    print(f"验证样本数: {info['val_size']}")
    print(f"测试样本数: {info['test_size']}")

    for x, y in train_loader:
        print(f"输入形状: {x.shape}, 目标形状: {y.shape}")
        break
