#!/usr/bin/env python3

import numpy as np

def generate_S_shape(n=200, epsilon=2.0, scaleX=50.0, scaleY=100.0):
    """
    生成大约 n 个围绕着 “S” 形曲线分布的点。
    参数:
      n       : 生成的点数
      epsilon : 在曲线上下的随机偏移量
      scaleX  : 控制 x 方向的大小 (S 横向幅度)
      scaleY  : 控制 y 方向的大小 (S 纵向幅度)
    返回:
      points  : (n, 2) 的 numpy 数组，每行 [x, y]
    """
    # 1. 参数化 t
    ts = np.linspace(-1, 1, n)
    
    # 2. 定义 S 曲线 (x(t), y(t))
    xs = scaleX * np.sin(np.pi * ts)
    ys = scaleY * ts
    
    # 3. 在每个点加上随机噪声 (此处为均匀分布)
    noise_x = np.random.uniform(-epsilon, epsilon, n)
    noise_y = np.random.uniform(-epsilon, epsilon, n)
    xs += noise_x
    ys += noise_y
    
    # 合并为 (n,2) 数组
    points = np.column_stack((xs, ys))
    return points

def main():
    # 生成 200 个点，偏移量为 ±3.0
    n = 200
    epsilon = 2
    
    # 得到数据点
    points = generate_S_shape(n=n, epsilon=epsilon, scaleX=50.0, scaleY=100.0)
    
    # 将结果写入 CSV 文件，每行两个浮点数，用空格分隔
    filename = "./examples/stipples/S_shape.csv"
    with open(filename, "w") as f:
        for (x, y) in points:
            f.write(f"{x:.3f} {y:.3f}\n")
    
    print(f"已生成约 {n} 个点，保存到 {filename}。")

if __name__ == "__main__":
    main()
