"""
展示.npy文件内容的工具脚本
用于查看引力波数据文件的结构和内容
"""

import numpy as np
import os
import sys

def show_npy_file(file_path):
    """
    展示.npy文件的内容
    
    参数:
        file_path: .npy文件路径
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在: {file_path}")
        return
    
    print(f"📁 文件路径: {file_path}")
    print(f"📊 文件大小: {os.path.getsize(file_path) / 1024:.2f} KB")
    print("-" * 60)
    
    # 加载数据
    data = np.load(file_path)
    
    # 基本信息
    print(f"📐 数据形状: {data.shape}")
    print(f"🔢 数据类型: {data.dtype}")
    print(f"📏 数据维度: {data.ndim}D")
    print(f"📦 总元素数: {data.size}")
    print("-" * 60)
    
    # 统计信息
    print("📈 统计信息:")
    print(f"  最小值: {data.min():.6f}")
    print(f"  最大值: {data.max():.6f}")
    print(f"  平均值: {data.mean():.6f}")
    print(f"  标准差: {data.std():.6f}")
    print("-" * 60)
    
    # 如果是3通道数据（引力波数据格式）
    if data.shape == (3, 4096):
        print("🌊 引力波数据格式 (3个探测器, 4096个采样点)")
        print("\n各探测器统计信息:")
        detector_names = ["LIGO Hanford", "LIGO Livingston", "Virgo"]
        for i, name in enumerate(detector_names):
            channel = data[i]
            print(f"  {name} (通道 {i}):")
            print(f"    最小值: {channel.min():.6f}")
            print(f"    最大值: {channel.max():.6f}")
            print(f"    平均值: {channel.mean():.6f}")
            print(f"    标准差: {channel.std():.6f}")
        print("-" * 60)
    
    # 显示数据的前几个值
    print("🔍 数据预览:")
    if data.ndim == 1:
        print(f"  前10个值: {data[:10]}")
        print(f"  后10个值: {data[-10:]}")
    elif data.ndim == 2:
        print(f"  第一行前10个值: {data[0, :10]}")
        print(f"  第一行后10个值: {data[0, -10:]}")
        if data.shape[0] > 1:
            print(f"  第二行前10个值: {data[1, :10]}")
        if data.shape[0] > 2:
            print(f"  第三行前10个值: {data[2, :10]}")
    
    print("-" * 60)
    
    # 尝试可视化（如果matplotlib可用）
    try:
        import matplotlib.pyplot as plt
        
        if data.shape == (3, 4096):
            # 绘制3个探测器的信号
            fig, axes = plt.subplots(3, 1, figsize=(12, 8))
            detector_names = ["LIGO Hanford", "LIGO Livingston", "Virgo"]
            
            for i, (ax, name) in enumerate(zip(axes, detector_names)):
                ax.plot(data[i], linewidth=0.5)
                ax.set_title(f"{name} - 通道 {i}")
                ax.set_xlabel("采样点")
                ax.set_ylabel("幅度")
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()  # 在Jupyter notebook中直接显示
        else:
            # 对于其他形状的数据，简单绘制
            plt.figure(figsize=(10, 6))
            if data.ndim == 1:
                plt.plot(data)
            elif data.ndim == 2:
                for i in range(min(3, data.shape[0])):
                    plt.plot(data[i], label=f"通道 {i}", alpha=0.7)
                plt.legend()
            plt.title(f"数据可视化: {os.path.basename(file_path)}")
            plt.xlabel("索引")
            plt.ylabel("值")
            plt.grid(True, alpha=0.3)
            plt.show()  # 在Jupyter notebook中直接显示
    except ImportError:
        print("💡 提示: 安装matplotlib可以生成可视化图表")
        print("   命令: pip install matplotlib")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python show_npy_file.py <npy_file_path>")
        print("\n示例:")
        print("  python show_npy_file.py /kaggle/input/g2net-gravitational-wave-detection/train/0/0/0/00005bced6.npy")
        print("  python show_npy_file.py data/raw/0/0/0/00005bced6.npy")
        return
    
    file_path = sys.argv[1]
    show_npy_file(file_path)


if __name__ == "__main__":
    main()

