"""
数据变换模块
将时域信号转换为时频表示（频谱图）用于CNN训练
使用恒定Q变换（CQT）而非标准STFT
"""

import torch
import torch.nn as nn
from nnAudio.Spectrogram import CQT1992v2
import torchaudio.transforms as T

class GWTransform(nn.Module):
    """
    引力波信号变换
    将时域信号转换为时频表示，使用CQT（恒定Q变换）
    """
    def __init__(self, sr=2048, fmin=20, fmax=500, hop_length=64):
        """
        初始化变换
        
        参数:
            sr: 采样率（Hz），默认2048
            fmin: 最小频率（Hz），默认20
            fmax: 最大频率（Hz），默认500
            hop_length: 跳跃长度，默认64
        """
        super().__init__()
        # CQT变换：使用对数频率尺度，适合引力波啁啾信号
        self.cqt = CQT1992v2(
            sr=sr,
            fmin=fmin,
            fmax=fmax,
            hop_length=hop_length,
            output_format="Magnitude",  # 输出幅度谱
            verbose=False
        )
        
        # 数据增强：时间掩码和频率掩码（仅在训练时使用）
        self.time_masking = T.TimeMasking(time_mask_param=5)   # 时间掩码
        self.freq_masking = T.FrequencyMasking(freq_mask_param=2)  # 频率掩码

    def forward(self, waveform, training=False):
        """
        前向传播：将时域信号转换为频谱图
        
        参数:
            waveform: 输入波形张量 (channels, samples) 或 (samples,)
            training: 是否为训练模式（影响数据增强）
            
        返回:
            spec: 频谱图张量 (channels, freq_bins, time_frames)
        """
        # 如果是一维输入，添加通道维度
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # CQT变换：时域 -> 时频域
        spec = self.cqt(waveform)
        # 对数缩放：增强动态范围，log1p(x) = log(1+x)，数值稳定
        spec = torch.log1p(spec)
        
        # 训练时应用数据增强
        if training:
            spec = self.time_masking(spec)   # 时间掩码增强
            spec = self.freq_masking(spec)   # 频率掩码增强
        
        return spec