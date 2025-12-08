"""
测试集预测模块
对测试集进行预测并生成submission.csv文件
"""

import os
import warnings
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm
from src.transforms import GWTransform

# 抑制警告
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


class TestDataset(Dataset):
    """
    测试数据集类
    用于加载测试集的.npy文件（不需要标签）
    """
    def __init__(self, file_paths):
        """
        初始化测试数据集
        
        参数:
            file_paths: 数据文件路径列表
        """
        self.file_paths = file_paths
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
            file_id: 文件ID（不含扩展名）
        """
        # 1. 加载数据
        path = self.file_paths[idx]
        waves = np.load(path)  # 形状: (3, 4096) - 3个探测器的时域信号
        
        # 2. 提取文件ID（从路径中提取，例如：00005bced6）
        file_id = os.path.splitext(os.path.basename(path))[0]
        
        # 3. 转换为张量
        wave_tensor = torch.from_numpy(waves).float()
        
        # 4. 安全归一化（每个通道独立进行最小-最大归一化）
        for i in range(3):  # 对3个探测器通道分别归一化
            w_min = wave_tensor[i].min()
            w_max = wave_tensor[i].max()
            wave_tensor[i] = (wave_tensor[i] - w_min) / (w_max - w_min + 1e-8)

        # 5. CQT变换和对数缩放（测试时不使用数据增强）
        image = self.transform(wave_tensor, training=False)
        
        # 6. 调整大小到224x224（EfficientNet标准输入尺寸）
        image = self.resize(image)
        
        # 7. 图像最终归一化（0到1范围）
        img_min = image.min()
        img_max = image.max()
        image = (image - img_min) / (img_max - img_min + 1e-8)
        
        return image, file_id


def find_test_files(test_dir):
    """
    扫描测试目录，找到所有.npy文件
    
    参数:
        test_dir: 测试目录路径
        
    返回:
        file_paths: 所有.npy文件的路径列表
        file_ids: 对应的文件ID列表
    """
    file_paths = []
    file_ids = []
    
    print(f"扫描测试目录: {test_dir}")
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]  # 提取文件ID（不含扩展名）
                file_paths.append(file_path)
                file_ids.append(file_id)
    
    print(f"找到 {len(file_paths)} 个测试文件")
    return file_paths, file_ids


def predict_test_set(model, test_loader, device):
    """
    对测试集进行预测
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        
    返回:
        predictions: 字典，键为文件ID，值为预测概率
    """
    model.eval()  # 设置为评估模式
    predictions = {}
    
    print("开始预测...")
    with torch.no_grad():  # 禁用梯度计算以节省内存
        for images, file_ids in tqdm(test_loader, desc="预测中"):
            images = images.to(device)
            
            # 前向传播
            outputs = model(images).squeeze()
            
            # 转换为概率（使用sigmoid）
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # 将概率转换为二分类标签（0或1）
            # 通常使用0.5作为阈值
            preds = (probs >= 0.5).astype(int)
            
            # 存储预测结果
            for file_id, pred in zip(file_ids, preds):
                predictions[file_id] = pred
    
    return predictions


def generate_submission(predictions, output_path="submission.csv"):
    """
    生成submission.csv文件
    
    参数:
        predictions: 字典，键为文件ID，值为预测标签（0或1）
        output_path: 输出文件路径
    """
    # 按文件ID排序（确保输出顺序一致）
    sorted_ids = sorted(predictions.keys())
    
    # 创建DataFrame
    df = pd.DataFrame({
        'id': sorted_ids,
        'target': [predictions[fid] for fid in sorted_ids]
    })
    
    # 保存为CSV文件
    df.to_csv(output_path, index=False)
    print(f"\n✅ 预测结果已保存到: {output_path}")
    print(f"共预测 {len(df)} 个样本")
    print(f"\n前5个预测结果预览:")
    print(df.head())


def main():
    """
    主函数：执行测试集预测流程
    """
    import argparse
    from src.model import GWClassifier
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试集预测')
    parser.add_argument('--test_dir', type=str, 
                       default='/kaggle/input/g2net-gravitational-wave-detection/test',
                       help='测试数据目录路径')
    parser.add_argument('--model_path', type=str,
                       default='models/best_model.pth',
                       help='模型权重文件路径')
    parser.add_argument('--output', type=str,
                       default='submission.csv',
                       help='输出CSV文件路径')
    parser.add_argument('--batch_size', type=int,
                       default=32,
                       help='批次大小')
    
    args = parser.parse_args()
    
    # 设置计算设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 使用设备: MacOS GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 使用设备: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用设备: CPU")
    
    # 1. 扫描测试文件
    print("\n[1/4] 扫描测试文件...")
    if not os.path.exists(args.test_dir):
        print(f"❌ 错误: 测试目录不存在: {args.test_dir}")
        return
    
    file_paths, file_ids = find_test_files(args.test_dir)
    
    if len(file_paths) == 0:
        print("❌ 错误: 未找到任何.npy文件")
        return
    
    # 2. 创建测试数据集和数据加载器
    print("\n[2/4] 创建测试数据集...")
    test_dataset = TestDataset(file_paths)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. 加载模型
    print(f"\n[3/4] 加载模型: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型文件不存在: {args.model_path}")
        return
    
    model = GWClassifier(pretrained=False)  # 不需要预训练权重，因为要加载已训练的权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # 4. 进行预测
    print("\n[4/4] 进行预测...")
    predictions = predict_test_set(model, test_loader, device)
    
    # 5. 生成submission.csv
    print("\n生成submission.csv文件...")
    generate_submission(predictions, args.output)
    
    print("\n✅ 预测完成！")


if __name__ == "__main__":
    main()

# 生成困难的时候可以尝试新建个tab