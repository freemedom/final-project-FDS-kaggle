"""
模型定义模块
使用EfficientNet-B0作为骨干网络的引力波分类器
"""

import torch
import torch.nn as nn
import timm

class GWClassifier(nn.Module):
    """
    引力波分类器
    使用EfficientNet-B0作为骨干网络的卷积神经网络
    将3个探测器的数据作为RGB通道处理
    """
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        """
        初始化模型
        
        参数:
            model_name: 模型名称（默认'efficientnet_b0'）
            pretrained: 是否使用ImageNet预训练权重
        """
        super().__init__()
        
        # 加载EfficientNet-B0（在ImageNet上预训练）
        # 标准输入：3通道（RGB），映射到我们的3个探测器
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,  # 使用预训练权重
            num_classes=0,          # 不包含分类头
            global_pool=''          # 不使用全局池化（我们自己实现）
        )
        
        num_features = self.backbone.num_features  # 获取特征维度
        
        # 分类头（带适度dropout）
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化到1x1
        self.head = nn.Sequential(
            nn.Flatten(),           # 展平
            nn.Dropout(p=0.2),       # Dropout正则化
            nn.Linear(num_features, 1)  # 二分类输出层
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像张量 (batch_size, 3, 224, 224)
            
        返回:
            output: 分类logits (batch_size, 1)
        """
        features = self.backbone(x)  # 提取特征
        pooled = self.global_pool(features)  # 全局池化
        output = self.head(pooled)  # 分类
        return output