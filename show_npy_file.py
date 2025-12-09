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
    
    注意: 在Jupyter notebook中使用时，建议先运行:
        %matplotlib inline
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
    # 使用科学计数法显示，避免极小值被四舍五入为0
    min_val = data.min()
    max_val = data.max()
    mean_val = data.mean()
    std_val = data.std()
    
    # 根据数值大小选择合适的显示格式
    if abs(min_val) < 1e-3 or abs(max_val) < 1e-3:
        # 对于极小的值，使用科学计数法
        print(f"  最小值: {min_val:.6e}")
        print(f"  最大值: {max_val:.6e}")
        print(f"  平均值: {mean_val:.6e}")
        print(f"  标准差: {std_val:.6e}")
    else:
        # 对于较大的值，使用普通格式
        print(f"  最小值: {min_val:.6f}")
        print(f"  最大值: {max_val:.6f}")
        print(f"  平均值: {mean_val:.6f}")
        print(f"  标准差: {std_val:.6f}")
    print("-" * 60)
    
    # 如果是3通道数据（引力波数据格式）
    if data.shape == (3, 4096):
        print("🌊 引力波数据格式 (3个探测器, 4096个采样点)")
        print("\n各探测器统计信息:")
        detector_names = ["LIGO Hanford", "LIGO Livingston", "Virgo"]
        for i, name in enumerate(detector_names):
            channel = data[i]
            min_val = channel.min()
            max_val = channel.max()
            mean_val = channel.mean()
            std_val = channel.std()
            
            print(f"  {name} (通道 {i}):")
            # 根据数值大小选择合适的显示格式
            if abs(min_val) < 1e-3 or abs(max_val) < 1e-3:
                print(f"    最小值: {min_val:.6e}")
                print(f"    最大值: {max_val:.6e}")
                print(f"    平均值: {mean_val:.6e}")
                print(f"    标准差: {std_val:.6e}")
            else:
                print(f"    最小值: {min_val:.6f}")
                print(f"    最大值: {max_val:.6f}")
                print(f"    平均值: {mean_val:.6f}")
                print(f"    标准差: {std_val:.6f}")
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
        import matplotlib
        import matplotlib.pyplot as plt
        
        # 在Jupyter notebook中，确保使用inline后端
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                # 在Jupyter中，使用inline后端
                ipython.run_line_magic('matplotlib', 'inline')
        except:
            # 如果不是在IPython环境中，尝试设置后端
            try:
                # 在Kaggle等环境中，可能需要使用Agg后端
                if 'KAGGLE' in os.environ or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
                    matplotlib.use('Agg')
            except:
                pass
        
        if data.shape == (3, 4096):
            # 绘制3个探测器的信号
            fig, axes = plt.subplots(3, 1, figsize=(12, 8))
            detector_names = ["LIGO Hanford", "LIGO Livingston", "Virgo"]
            
            for i, (ax, name) in enumerate(zip(axes, detector_names)):
                ax.plot(data[i], linewidth=0.5)
                ax.set_title(f"{name} - Channel {i}")
                ax.set_xlabel("Sample Point")
                ax.set_ylabel("Amplitude")
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            # 在Jupyter notebook中显示图表     #图还是不显示
            try:
                from IPython.display import display
                display(plt.gcf())
            except:
                plt.show()
        else:
            # 对于其他形状的数据，简单绘制
            plt.figure(figsize=(10, 6))
            if data.ndim == 1:
                plt.plot(data)
            elif data.ndim == 2:
                for i in range(min(3, data.shape[0])):
                    plt.plot(data[i], label=f"Channel {i}", alpha=0.7)
                plt.legend()
            plt.title(f"Data Visualization: {os.path.basename(file_path)}")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            # 在Jupyter notebook中显示图表
            try:
                from IPython.display import display
                display(plt.gcf())
            except:
                plt.show()
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

