"""
引力波检测项目 - 主训练脚本
使用EfficientNet-B0模型训练引力波信号分类器
"""

import os
import warnings
import random
import numpy as np

# --- 1. 警告抑制 ---
# 抑制各种警告信息，保持输出清洁
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pin_memory.*") 

# --- 2. 导入库 ---
import torch
import matplotlib.pyplot as plt
from src.dataset import create_dataloaders
from src.model import GWClassifier
from src.train import Trainer

# --- 配置参数 ---
# DATA_DIR = "data/raw"  # 数据目录
DATA_DIR = "E:/老笔记本电脑移出/data..raw"
LABELS_FILE = os.path.join(DATA_DIR, "subset_labels.csv")  # 标签文件路径
MODEL_SAVE_PATH = "models/best_model.pth"  # 模型保存路径

BATCH_SIZE = 32  # 批次大小
EPOCHS = 12      # 训练轮数
LEARNING_RATE = 5e-5  # 学习率
SEED = 42        # 随机种子，确保结果可复现         

def set_seed(seed):
    """
    设置所有随机数生成器的种子，确保结果可复现
    
    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA的确定性行为（可能会稍微减慢速度，但值得）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed}")

def plot_results(history):
    """
    绘制训练历史结果图表
    
    参数:
        history: 包含训练和验证损失及AUC的字典
    """
    plt.figure(figsize=(12, 5))
    
    # 图表1: 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss', linestyle='--')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 图表2: AUC分数曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_auc'], label='Train AUC')
    plt.plot(history['val_auc'], label='Val AUC', linestyle='--')
    plt.title('AUC Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_results.png")
    print("\n📊 Plots saved as 'training_results.png'")

def main():
    """
    主函数：执行完整的训练流程
    """
    # 0. 设置随机种子（首先执行以确保可复现性）
    set_seed(SEED)

    # 1. 设置计算设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Device: MacOS GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using Device: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using Device: CPU")

    # 2. 准备数据
    print("\n[1/3] Loading Data (Smart Search)...")
    try:
        train_loader, val_loader = create_dataloaders(
            data_dir=DATA_DIR,
            labels_file=LABELS_FILE,
            batch_size=BATCH_SIZE
        )
        print(f"Data loaded successfully.")
        print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 3. 初始化模型
    print("\n[2/3] Initializing EfficientNet Model (RGB Mode)...")
    model = GWClassifier(pretrained=True)  # 使用ImageNet预训练权重
    
    # 4. 开始训练
    print("\n[3/3] Starting Training Loop...")
    os.makedirs("models", exist_ok=True)  # 创建模型保存目录
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=LEARNING_RATE
    )
    
    history = trainer.fit(epochs=EPOCHS, save_path=MODEL_SAVE_PATH)
    
    # 5. 收尾工作：绘制结果图表
    plot_results(history)
    print(f"\n✅ Training Complete! Best model weights saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()