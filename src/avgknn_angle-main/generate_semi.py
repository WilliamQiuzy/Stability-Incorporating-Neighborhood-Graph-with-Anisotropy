#!/usr/bin/env python3
import numpy as np

def generate_semicircle_clusters(
    M=6,
    points_per_cluster=30,
    R=100.0,
    ellipse_axes=(10, 4),
    random_seed=42
):
    """
    在上半圆弧 (theta in [0, pi]) 上生成 M 个聚簇，每个聚簇含 points_per_cluster 个点，
    聚簇呈椭圆分布，椭圆主轴长度由 ellipse_axes 控制。
    
    参数：
      M                : 聚簇数
      points_per_cluster : 每个聚簇的点数
      R                : 半径
      ellipse_axes     : (a, b) 椭圆半轴长度，a 方向为弧线切向，b 方向为弧线法向
      random_seed      : 随机种子，便于复现结果
    
    返回：
      points           : (N, 2) 的 numpy 数组，N = M * points_per_cluster
    """
    np.random.seed(random_seed)
    
    # 生成 M 个等间隔的角度（不含 0 与 pi 的极端，避免两端聚簇过于集中）
    thetas = np.linspace(0.0, np.pi, M+2)[1:-1]
    
    # 保存所有点的列表
    all_points = []
    
    for theta in thetas:
        # 1. 计算该聚簇中心在弧线上的坐标
        center_x = R * np.cos(theta)
        center_y = R * np.sin(theta)
        
        # 2. 在局部坐标系中采样 (u, v) ~ N(0, I)
        #    然后通过变换将其映射到真实坐标
        #    - u 方向：沿弧线切向
        #    - v 方向：沿弧线法向
        #    椭圆轴长度由 ellipse_axes=(a, b) 给定
        
        # 弧线切向 (tx, ty) 与法向 (nx, ny)
        #   切向：与圆心相连向量旋转 90 度
        #   法向：指向圆心->弧点方向
        #   注意：法向与切向需单位化
        nx, ny = center_x / R, center_y / R  # 指向弧线上点的法向 (单位向量)
        # 将 (nx, ny) 逆时针旋转 90 度得到切向 (tx, ty)
        tx, ty = -ny, nx
        
        # 采样点
        for _ in range(points_per_cluster):
            # 先在局部椭圆中采样
            u, v = np.random.normal(0, 1, 2)
            # 放缩到椭圆半轴长度
            u_scaled = u * ellipse_axes[0]
            v_scaled = v * ellipse_axes[1]
            
            # 转换到全局坐标： center + u_scaled*(tx, ty) + v_scaled*(nx, ny)
            px = center_x + u_scaled * tx + v_scaled * nx
            py = center_y + u_scaled * ty + v_scaled * ny
            
            all_points.append((px, py))
    
    return np.array(all_points)

def main():
    # 示例参数
    M = 9  # 聚簇数量
    points_per_cluster = 15
    R = 100.0
    ellipse_axes = (20, 2)
    
    points = generate_semicircle_clusters(
        M=M,
        points_per_cluster=points_per_cluster,
        R=R,
        ellipse_axes=ellipse_axes,
        random_seed=42
    )
    
    # 将结果写入文件
    filename = "./examples/stipples/semicircle_clusters.csv"
    with open(filename, "w") as f:
        for (x, y) in points:
            f.write(f"{x:.3f} {y:.3f}\n")
    
    print(f"生成了 {len(points)} 个点，写入 {filename}。")

if __name__ == "__main__":
    main()
