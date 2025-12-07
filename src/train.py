"""
训练模块
包含完整的训练循环、验证、梯度裁剪、学习率调度等功能
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import time 

class Trainer:
    """
    引力波分类器训练器
    包含：训练循环、验证、梯度裁剪、学习率调度和计时器
    """
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 计算设备（CPU/GPU）
            lr: 学习率
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数：二元交叉熵（带logits，数值稳定）
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 优化器：AdamW是EfficientNet的标准选择
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        
        # 指标和历史记录
        self.best_score = 0.0  # 最佳验证AUC分数
        self.history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    def train_one_epoch(self):
        """
        训练一个epoch
        
        返回:
            epoch_loss: 平均训练损失
            epoch_auc: 训练集AUC分数
        """
        self.model.train()  # 设置为训练模式
        running_loss = 0.0
        all_targets = []
        all_preds = []
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            # 1. 清零梯度
            self.optimizer.zero_grad()
            
            # 2. 前向传播
            outputs = self.model(images).squeeze()
            loss = self.criterion(outputs, targets)
            
            # 3. 反向传播
            loss.backward()
            
            # 4. 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 5. 优化器更新
            self.optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy()  # 转换为概率
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(self.train_loader)
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except:
            epoch_auc = 0.5  # 如果计算失败，返回随机猜测的AUC
            
        return epoch_loss, epoch_auc

    def evaluate(self):
        """
        在验证集上评估模型
        
        返回:
            avg_loss: 平均验证损失
            auc_score: 验证集AUC分数
        """
        self.model.eval()  # 设置为评估模式
        running_loss = 0.0
        all_targets = []
        all_preds = []
        
        with torch.no_grad():  # 禁用梯度计算以节省内存和加速
            for images, targets in tqdm(self.val_loader, desc="Validation", leave=False):
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = running_loss / len(self.val_loader)
        try:
            auc_score = roc_auc_score(all_targets, all_preds)
        except:
            auc_score = 0.5  # 如果计算失败，返回随机猜测的AUC
            
        return avg_loss, auc_score

    def fit(self, epochs, save_path="models/best_model.pth"):
        """
        执行完整的训练流程
        
        参数:
            epochs: 训练轮数
            save_path: 模型保存路径
            
        返回:
            history: 包含训练历史的字典
        """
        print(f"Starting training on {self.device}...")
        
        # 学习率调度器：当验证AUC不再提升时降低学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2  # 耐心值2，因子0.5
        )
        
        total_start = time.time()  # 开始总计时器
        
        for epoch in range(epochs):
            epoch_start = time.time()  # 开始epoch计时器
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练和验证
            train_loss, train_auc = self.train_one_epoch()
            val_loss, val_auc = self.evaluate()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            # 更新学习率
            scheduler.step(val_auc)
            
            # 计算epoch时间
            epoch_end = time.time()
            epoch_mins = int((epoch_end - epoch_start) / 60)
            epoch_secs = int((epoch_end - epoch_start) % 60)
            
            print(f"⏱️ Time: {epoch_mins}m {epoch_secs}s")
            print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
            
            # 保存最佳模型
            if val_auc > self.best_score:
                print(f"🚀 Score Improved ({self.best_score:.4f} -> {val_auc:.4f}). Saving model...")
                self.best_score = val_auc
                torch.save(self.model.state_dict(), save_path)
            else:
                print("Score did not improve.")
        
        # 计算总训练时间
        total_end = time.time()
        total_mins = int((total_end - total_start) / 60)
        total_secs = int((total_end - total_start) % 60)
        print(f"\n🏁 Total Training Time: {total_mins}m {total_secs}s")
        
        return self.history