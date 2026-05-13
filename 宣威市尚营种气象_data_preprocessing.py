#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本：读取宣威市尚营种气象Excel文件，整合到单个CSV
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def process_xuanwei_file(file_path):
    """
    处理宣威市尚营种气象单个Excel文件，提取气象数据

    Args:
        file_path: Excel文件路径

    Returns:
        DataFrame: 包含时间、温度、湿度、光照强度、风向、风速、降雨量的数据
    """
    try:
        # 读取Excel文件
        df_raw = pd.read_excel(file_path)

        # 获取文件名作为标识
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # 提取需要的列：收集时间、大气温度、大气湿度、光照强度、风向、风速、降雨量
        target_cols = ['收集时间', '大气温度', '大气湿度', '光照强度', '风向', '风速', '降雨量']

        # 检查列是否存在
        if not all(col in df_raw.columns for col in target_cols):
            print(f"警告: 文件 {file_name} 缺少必要列")
            return pd.DataFrame()

        # 提取目标列
        data_df = df_raw[target_cols].copy()

        # 重命名列为简洁名称
        data_df.columns = ['时间', '温度', '湿度', '光照强度', '风向', '风速', '降雨量']

        # 转换时间列为datetime（保留原始时间，不做修改）
        data_df['时间'] = pd.to_datetime(data_df['时间'], errors='coerce')

        # 数值列转换
        numeric_cols = ['温度', '湿度', '光照强度', '风向', '风速', '降雨量']
        for col in numeric_cols:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

        # 去除无效数据（时间和温度湿度至少要有效）
        data_df = data_df.dropna(subset=['时间', '温度', '湿度'])

        print(f"  文件 {file_name}: {len(data_df)} 条有效数据")

        return data_df

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return pd.DataFrame()


def main():
    """
    主函数：处理所有Excel文件，按时间分组计算平均值，保存结果
    """
    print("开始数据预处理...")

    # 定义输入输出目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir_name = "宣威市尚营种气象"
    input_dir = os.path.join(current_dir, input_dir_name)
    output_dir = os.path.join(current_dir, "天气数据", input_dir_name)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 获取所有Excel文件
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在 - {input_dir}")
        return

    excel_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]

    if not excel_files:
        print("未找到Excel文件")
        return

    print(f"找到 {len(excel_files)} 个Excel文件")

    # 存储所有处理后的数据
    all_data_frames = []

    # 处理每个文件
    print("\n正在处理文件...")
    for filename in excel_files:
        file_path = os.path.join(input_dir, filename)
        data_df = process_xuanwei_file(file_path)

        if not data_df.empty:
            all_data_frames.append(data_df)

    if not all_data_frames:
        print("未处理任何有效数据")
        return

    # 合并所有数据
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"\n合并后总数据条数: {len(combined_df)}")

    # 按原始时间分组计算平均值（不修改时间）
    print("正在按原始时间分组计算平均值...")
    grouped = combined_df.groupby('时间').agg({
        '温度': 'mean',
        '湿度': 'mean',
        '光照强度': 'mean',
        '风向': 'mean',
        '风速': 'mean',
        '降雨量': 'mean'
    }).reset_index()

    # 重命名列
    grouped.columns = ['时间', '平均温度', '平均湿度', '平均光照强度', '平均风向', '平均风速', '平均降雨量']

    # 平均值保留两位小数
    grouped['平均温度'] = grouped['平均温度'].round(2)
    grouped['平均湿度'] = grouped['平均湿度'].round(2)
    grouped['平均光照强度'] = grouped['平均光照强度'].round(2)
    grouped['平均风向'] = grouped['平均风向'].round(2)
    grouped['平均风速'] = grouped['平均风速'].round(2)
    grouped['平均降雨量'] = grouped['平均降雨量'].round(2)

    # 按时间排序
    grouped = grouped.sort_values('时间').reset_index(drop=True)

    # 添加序号列
    grouped.insert(0, '序号', range(1, len(grouped) + 1))

    # 保存结果（CSV文件名使用读取的文件夹名称）
    output_path = os.path.join(output_dir, f'{input_dir_name}.csv')
    grouped.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n结果已保存到: {output_path}")

    # 打印统计信息
    print("\n统计信息:")
    print(f"总数据条数（原始）: {len(combined_df)}")
    print(f"时间点数（去重后）: {len(grouped)}")
    print(f"时间范围: {grouped['时间'].min()} 至 {grouped['时间'].max()}")
    print(f"总体平均温度: {grouped['平均温度'].mean():.2f}°C")
    print(f"总体平均湿度: {grouped['平均湿度'].mean():.2f}%")
    print(f"总体平均光照强度: {grouped['平均光照强度'].mean():.2f}")
    print(f"总体平均风向: {grouped['平均风向'].mean():.2f}°")
    print(f"总体平均风速: {grouped['平均风速'].mean():.2f}m/s")
    print(f"总体平均降雨量: {grouped['平均降雨量'].mean():.2f}mm")

    print("\n数据预处理完成!")


if __name__ == "__main__":
    main()
