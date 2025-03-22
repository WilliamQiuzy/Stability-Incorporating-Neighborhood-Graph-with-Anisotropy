#!/usr/bin/env python3
import numpy as np
import csv

def generate_new_core_points(core_points, flagged_indices):
    """
    在指定斜边上增加一个新的核心点（取中点），并生成新的核心点列表。
    参数:
      core_points: 原始核心点（按顺序组成闭合曲线）
      flagged_indices: 需要在其起始边上增加中点的核心点索引集合
    返回:
      新的核心点数组（包含原始点和新增中点）
    """
    new_core_points = []
    num_orig = len(core_points)
    for i in range(num_orig):
        new_core_points.append(core_points[i])
        # 计算下一个点（闭合曲线处理）
        j = (i + 1) % num_orig
        # 如果当前段需要增加核心点，则在此段中插入中点
        if i in flagged_indices:
            midpoint = (core_points[i] + core_points[j]) / 2.0
            new_core_points.append(midpoint)
    return np.array(new_core_points)

def generate_interpolated_points(new_core_points, high_density_segments, high_density_count, low_density_count):
    """
    在每个核心点段上生成插值点，部分段使用高密度，其余段使用低密度。
    参数:
      new_core_points: 核心点数组（闭合曲线）
      high_density_segments: 需要高密度采样的段索引集合
      high_density_count: 高密度段的插值点数量（不包含段尾）
      low_density_count: 低密度段的插值点数量（不包含段尾）
    返回:
      所有插值点的列表，每个点为 [x, y] 格式
    """
    interpolated_points = []
    num_points = len(new_core_points)
    for i in range(num_points):
        start = new_core_points[i]
        end = new_core_points[(i + 1) % num_points]
        # 根据当前段是否为高密度段，选择对应的插值点数量
        count = high_density_count if i in high_density_segments else low_density_count
        for k in range(count):
            t = k / float(count)
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            interpolated_points.append([x, y])
    return interpolated_points

def write_points_to_file(points, filename):
    """
    将点写入文件，每行两个浮点数，用空格分隔，保留三位小数。
    """
    with open(filename, "w", newline='') as csvfile:
        # 指定分隔符为空格
        writer = csv.writer(csvfile, delimiter=' ')
        for point in points:
            writer.writerow([f"{point[0]:.3f}", f"{point[1]:.3f}"])
    print(f"Generated {len(points)} points, saved in {filename}")

if __name__ == "__main__":
    # 原始核心点（可根据需求进一步调整，以获得合适形状）
    core_points = np.array([
        [0.0, 8.0],    # 点1：小突起顶点
        [5.0, 5.0],    # 点2：小突起基座右端
        [10.0, 0.0],   # 点3：右侧过渡点
        [8.0, -5.0],   # 点4：大突起基座右端
        [0.0, -8.0],   # 点5：大突起顶点
        [-8.0, -5.0],  # 点6：大突起基座左端
        [-10.0, 0.0],  # 点7：左侧过渡点
        [-5.0, 5.0]    # 点8：小突起基座左端
    ])
    
    # 在以下边上增加额外核心点（斜边：1-2, 4-5, 5-6, 8-1，对应索引0, 3, 4, 7）
    flagged_indices = {0, 3, 4, 7}

    # 生成新的核心点（包含新增中点）
    new_core_points = generate_new_core_points(core_points, flagged_indices)

    # 设置不同段的密度：以下段索引采用高密度，其余段低密度
    # （这里高密度段为：0, 3, 4, 7, 8, 10, 11；具体根据核心点调整）
    high_density_segments = {0, 3, 4, 7, 8, 10, 11}
    high_density_count = 50   # 高密度段插值点数
    low_density_count = 30    # 低密度段插值点数

    # 生成所有插值点
    points = generate_interpolated_points(new_core_points, high_density_segments, high_density_count, low_density_count)

    # 写入文件，文件中每行两个浮点数，使用空格分隔（无逗号）
    output_filename = "./examples/stipples/new_core_points.csv"
    write_points_to_file(points, output_filename)
