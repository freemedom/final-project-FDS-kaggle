"""
数据集模块
处理引力波时间序列数据的加载和预处理
"""

import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from src.transforms import GWTransform

class G2NetDataset(Dataset):
    """
    G2Net数据集类
    将引力波时间序列数据转换为图像格式用于CNN训练
    """
    def __init__(self, file_paths, targets, training=False):
        """
        初始化数据集
        
        参数:
            file_paths: 数据文件路径列表
            targets: 标签列表（0=噪声，1=引力波信号）
            training: 是否为训练模式（影响数据增强）
        """
        self.file_paths = file_paths
        self.targets = targets
        self.training = training
        self.transform = GWTransform()  # CQT变换
        self.resize = Resize((224, 224), antialias=True)  # 调整到EfficientNet标准输入尺寸

    def __len__(self):
        """返回数据集大小"""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        参数:
            idx: 样本索引
            
        返回:
            image: 处理后的图像张量 (3, 224, 224)
            label: 标签 (0或1)
        """
        # 1. 加载数据
        path = self.file_paths[idx]
        waves = np.load(path)  # 形状: (3, 4096) - 3个探测器的时域信号
        
        # 2. 转换为张量
        wave_tensor = torch.from_numpy(waves).float()
        
        # 3. 安全归一化（每个通道独立进行最小-最大归一化）
        # 为什么需要归一化：
        # - 不同探测器的信号幅度可能差异很大（LIGO Hanford、LIGO Livingston、Virgo）
        # - 归一化可以消除幅度差异，让模型关注信号形状而非绝对强度
        # - 独立归一化每个通道：保持各探测器的相对特征，避免强信号通道主导弱信号通道
        for i in range(3):  # 遍历3个探测器通道（LIGO Hanford、LIGO Livingston、Virgo）
            w_min = wave_tensor[i].min()  # 获取第i个通道的最小值
            w_max = wave_tensor[i].max()  # 获取第i个通道的最大值
            # 最小-最大归一化公式：(x - min) / (max - min)
            # 结果范围：[0, 1]，其中0对应最小值，1对应最大值
            # 添加1e-8：防止max==min时除零错误（当信号为常数时）
            wave_tensor[i] = (wave_tensor[i] - w_min) / (w_max - w_min + 1e-8)

        # 4. CQT变换和对数缩放
        image = self.transform(wave_tensor, training=self.training)
        
        # 5. 调整大小到224x224（EfficientNet标准输入尺寸）
        image = self.resize(image)
        
        # 6. 图像最终归一化（0到1范围）
        img_min = image.min()
        img_max = image.max()
        image = (image - img_min) / (img_max - img_min + 1e-8)
        
        label = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, label

def create_dataloaders(data_dir, labels_file, batch_size=32, split_ratio=0.8):
    """
    创建训练和验证数据加载器
    
    参数:
        data_dir: 数据目录路径
        labels_file: 标签CSV文件路径
        batch_size: 批次大小
        split_ratio: 训练集比例（默认0.8，即80%训练，20%验证）
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    print(f"Scanning {data_dir} for .npy files...")
    df = pd.read_csv(labels_file)
    
    # 打印出文件数量
    print(f"Found {len(df)} samples in labels file.")
    
    # 扫描目录，建立文件ID到文件路径的映射
    file_path_map = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".npy"):
                file_id = os.path.splitext(file)[0]  # 提取文件ID（不含扩展名）
                file_path_map[file_id] = os.path.join(root, file)
    
    # 打印出file_path_map的数量
    print(f"Found {len(file_path_map)} files in data directory.")
    
    # 匹配标签文件中的ID和实际数据文件
    file_paths = []
    valid_indices = []
    for idx, row in df.iterrows():
        f_id = row['id']
        if f_id in file_path_map:
            file_paths.append(file_path_map[f_id])
            valid_indices.append(idx)
            
    targets = df.loc[valid_indices, 'target'].values
    print(f"Matched {len(file_paths)} samples.")
    
    if len(file_paths) == 0: 
        raise FileNotFoundError("No files found.")

    # 划分训练集和验证集
    dataset_size = len(file_paths)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    train_idx, val_idx = indices[:split], indices[split:]
    
    # 创建数据集对象
    train_dataset = G2NetDataset([file_paths[i] for i in train_idx], [targets[i] for i in train_idx], training=True)
    val_dataset = G2NetDataset([file_paths[i] for i in val_idx], [targets[i] for i in val_idx], training=False)
    
    # 创建数据加载器（标准配置，不显式设置num_workers或pin_memory）
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader