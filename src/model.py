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
                - EfficientNet 族共有 B0~B7 等版本，B0 为最小/基线模型
                - 结构拆解：
                    • MBConv（Mobile Inverted Bottleneck）：先升维再深度可分离卷积、再降维的倒残差块，计算量低且表达力强
                    • SE（Squeeze-and-Excitation）注意力：自适应为每个通道分配权重，突出关键信号、抑制冗余
                    • Swish/SiLU 激活：平滑的非线性激活，相比 ReLU 在梯度和收敛性上更友好
                    • 多个 MBConv+SE 块按阶段堆叠，并配合复合缩放调整宽度/深度/分辨率
                - B0 的复合缩放系数最小：宽度/深度/分辨率几乎不放大，参数量/FLOPs 全系最低，默认输入 224x224
                - 对比 B7：放大倍率最高——
                    • 更宽：卷积层的通道数（filters）显著增加，例如首层由 ~32 提升到 ~64+，同层能提取更多特征通道
                    • 更深：MBConv 模块数量/重复次数更多，同一阶段会堆叠更多块，提取更复杂的层级特征
                    • 更高分辨率：输入 600x600，特征图尺寸更大，计算量成倍增长
                  因此 B7 的参数量和 FLOPs 远高于 B0
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