"""
数据变换模块
将时域信号转换为时频表示（频谱图）用于CNN训练
使用恒定Q变换（CQT）而非标准STFT
"""

import torch
import torch.nn as nn
# CQT1992v2: 恒定Q变换（Constant-Q Transform）的1992年算法实现（nnAudio库的v2版本）
# - CQT是一种时频变换方法，使用对数频率尺度（logarithmic frequency scale）
# - 与标准STFT（短时傅里叶变换）不同，CQT在不同频率使用不同的时间分辨率
# - 低频使用长窗口（高频率分辨率），高频使用短窗口（高时间分辨率）
# - 特别适合分析引力波啁啾信号（频率随时间快速变化的信号）
# - 1992年由Brown和Puckette提出，是音乐信号分析和音频处理中的经典方法
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
            sr: 采样率（Sampling Rate，单位：Hz），默认2048
                - 表示每秒采集的信号样本数
                - 2048 Hz 意味着每秒记录2048个数据点
                - 对于4096个样本的信号，时间长度为 4096/2048 = 2秒
                - 根据奈奎斯特定理，可分析的最高频率为 sr/2 = 1024 Hz
            
            fmin: 最小频率（单位：Hz），默认20
                - CQT变换分析的频率范围下限
                - 20 Hz 是引力波信号的重要频率范围起点
                - 低于此频率的信号将被忽略
            
            fmax: 最大频率（单位：Hz），默认500
                - CQT变换分析的频率范围上限
                - 500 Hz 覆盖了引力波合并事件的主要频率范围
                - 高于此频率的信号将被忽略
            
            hop_length: 跳跃长度（单位：样本数），默认64
                - 在时间轴上移动窗口的步长
                - 控制频谱图的时间分辨率：hop_length越小，时间分辨率越高
                - 对于4096个样本，hop_length=64 会产生约 4096/64 = 64个时间帧
                - 较小的值：时间分辨率高但计算量大；较大的值：计算快但时间分辨率低
        """
        super().__init__()
        # CQT1992v2变换：使用对数频率尺度，适合引力波啁啾信号
        # CQT（恒定Q变换）的优势：
        # 1. 对数频率尺度：更符合人耳和物理信号的感知特性
        # 2. 自适应分辨率：低频高频率分辨率，高频高时间分辨率
        # 3. 适合啁啾信号：引力波信号频率随时间快速变化，CQT能更好地捕捉这种特征
        # 4. 相比STFT：在分析宽频带信号时提供更好的时频表示
        self.cqt = CQT1992v2(
            sr=sr,
            fmin=fmin,
            fmax=fmax,
            hop_length=hop_length,
            output_format="Magnitude",  # 输出幅度谱（只保留幅度信息，忽略相位）
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