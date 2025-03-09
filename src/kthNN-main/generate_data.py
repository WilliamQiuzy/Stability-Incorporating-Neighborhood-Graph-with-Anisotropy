#!/usr/bin/env python3

import numpy as np

def generate_interpolated_points(p1, p2, num_points):
    x1, y1 = p1
    x2, y2 = p2
    points = []
    for i in range(num_points):
        t = i / num_points   # t从0到(1 - 1/num_points)
        x = x1 + t*(x2 - x1)
        y = y1 + t*(y2 - y1)
        points.append((x, y))
    return points

def generate_protruded_square_approx(filename="protruded_square.csv"):
    
    corners = [
            (0, 0),
            (300, 0),
            (240, 300),
            (200, 340),  # 大突起顶点
            (160, 300),
            (130, 315),  # 小突起顶点
            (100, 300),
            (0, 300),
            (0, 0)       # 闭合
        ]
    
    # 每条线段插值的数量(不含该段最后一个拐点,以免下条线段再重复起点)
    num_points_per_segment = 63
    
    all_points = []
    
    for i in range(len(corners) - 1):
        p_start = corners[i]
        p_end   = corners[i+1]
        segment_pts = generate_interpolated_points(p_start, p_end, num_points_per_segment)
        all_points.extend(segment_pts)
    
    # 最后一个拐点 corners[-1] 可视情况添加:
    # all_points.append(corners[-1])
    # 这样就可以保证形状闭合.
    
    # 将结果写入文件(空格分隔)
    with open(filename, "w") as f:
        for (x, y) in all_points:
            f.write(f"{x:.3f} {y:.3f}\n")  # 保留三位小数
    
    print(f"已生成大约 {len(all_points)} 个点，保存在 {filename}。")

if __name__ == "__main__":
    generate_protruded_square_approx("./examples/stipples/protruded_square.csv")
