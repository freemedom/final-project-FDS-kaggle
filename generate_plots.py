"""
引力波检测项目 - 模型评估和可视化脚本
生成混淆矩阵、ROC曲线和检测到的信号画廊
"""

import os
import warnings
import random
import numpy as np

# --- 警告抑制 ---
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pin_memory.*") 

# --- 导入库 ---
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.dataset import create_dataloaders
from src.model import GWClassifier
from tqdm import tqdm

# --- 配置参数 ---
DATA_DIR = "data/raw"  # 数据目录
LABELS_FILE = os.path.join(DATA_DIR, "subset_labels.csv")  # 标签文件路径
MODEL_PATH = "models/best_model.pth"  # 模型路径
SEED = 42  # 必须与训练时使用的种子相同，以确保数据划分一致

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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed}")

def generate_advanced_plots():
    """
    生成高级评估图表：混淆矩阵、ROC曲线和信号画廊
    """
    # 设置随机种子（对数据划分一致性至关重要）
    set_seed(SEED)
    
    # 设置计算设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Device: MacOS GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using Device: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using Device: CPU")

    print(f"🚀 Loading Best Model from {MODEL_PATH}...")
    
    # 加载数据（仅验证集）
    # 注意：create_dataloaders将使用相同的种子重新创建相同的80/20划分
    _, val_loader = create_dataloaders(DATA_DIR, LABELS_FILE, batch_size=32)
    
    # 加载模型
    model = GWClassifier(pretrained=False)  # 不需要预训练权重，因为要加载已训练的权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 运行推理
    print("🔍 Running Inference on Validation Set...")
    all_preds = []  # 存储所有预测概率
    all_targets = []  # 存储所有真实标签
    top_hits = []  # 存储高置信度的检测结果
    
    with torch.no_grad():  # 禁用梯度计算以节省内存
        for images, targets in tqdm(val_loader):
            images = images.to(device)
            outputs = model(images).squeeze()  # 模型输出
            preds = torch.sigmoid(outputs).cpu().numpy()  # 转换为概率
            
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
            # 保存最佳检测结果用于画廊展示
            for i, p in enumerate(preds):
                if targets[i] == 1 and p > 0.9:  # 真实信号且置信度>90%
                    top_hits.append((p, images[i].cpu(), targets[i]))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # --- 图表1: 混淆矩阵 ---
    binary_preds = (all_preds > 0.5).astype(int)  # 将概率转换为二分类预测
    cm = confusion_matrix(all_targets, binary_preds)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Noise', 'GW Signal'],
                yticklabels=['Noise', 'GW Signal'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('plot_confusion_matrix.png')
    print("✅ Saved plot_confusion_matrix.png")
    
    # --- 图表2: ROC曲线 ---
    fpr, tpr, _ = roc_curve(all_targets, all_preds)  # 计算ROC曲线
    roc_auc = auc(fpr, tpr)  # 计算AUC值
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 随机分类器基准线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_roc_curve.png')
    print("✅ Saved plot_roc_curve.png")

    # --- 图表3: 信号画廊 ---
    top_hits.sort(key=lambda x: x[0], reverse=True)  # 按置信度降序排序
    best_6 = top_hits[:6]  # 选择前6个最佳检测结果
    
    if len(best_6) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Top Detected Gravitational Waves (Confidence > 90%)", fontsize=16)
        
        for idx, (conf, img_tensor, target) in enumerate(best_6):
            if idx >= 6: break
            row = idx // 3  # 计算行索引
            col = idx % 3   # 计算列索引
            
            # 显示第1个通道（LIGO Hanford探测器）
            img_display = img_tensor[0].numpy()
            
            ax = axes[row, col]
            im = ax.imshow(img_display, origin='lower', aspect='auto', cmap='inferno')
            ax.set_title(f"Confidence: {conf*100:.2f}%")
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig('plot_galaxy_gallery.png')
        print("✅ Saved plot_galaxy_gallery.png")
    else:
        print("⚠️ No high-confidence hits found.")

if __name__ == "__main__":
    generate_advanced_plots()