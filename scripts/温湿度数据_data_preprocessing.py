#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本：读取多个Excel文件，按时间分组计算平均值
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def process_excel_file(file_path):
    """
    处理单个Excel文件，提取测量数据
    
    Args:
        file_path: Excel文件路径
        
    Returns:
        DataFrame: 包含日期、温度、湿度的数据
    """
    try:
        # 读取Excel文件
        df_raw = pd.read_excel(file_path)
        
        # 获取文件名作为标识
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 提取数据部分（尝试自动检测表头）
        data_df = pd.DataFrame()
        
        # 尝试方式1：查找包含"序号"的表头行
        header_row = None
        for i in range(min(30, len(df_raw))):
            second_col = str(df_raw.iloc[i, 1]).strip() if len(df_raw.columns) > 1 else ''
            if second_col == '序号':
                header_row = i
                break
        
        if header_row is not None:
            # 建立表头映射
            col_map = {}
            for j in range(df_raw.shape[1]):
                cell = str(df_raw.iloc[header_row, j]).strip() if pd.notna(df_raw.iloc[header_row, j]) else ''
                cell_clean = cell.replace('℃', '').replace('%', '').strip()
                if cell_clean:
                    col_map[cell_clean] = j
            
            # 查找温度、湿度、日期列
            temp_col = None
            hum_col = None
            date_col = None
            for key, idx in col_map.items():
                if '温度' in key:
                    temp_col = idx
                elif '湿度' in key:
                    hum_col = idx
                elif '日期' in key or '时间' in key:
                    date_col = idx
            
            if temp_col is not None and hum_col is not None:
                data_df = pd.DataFrame({
                    '日期': pd.to_datetime(df_raw.iloc[header_row+1:, date_col], errors='coerce') if date_col else None,
                    '温度': pd.to_numeric(df_raw.iloc[header_row+1:, temp_col], errors='coerce'),
                    '湿度': pd.to_numeric(df_raw.iloc[header_row+1:, hum_col], errors='coerce'),
                    '来源文件': file_name
                })
        else:
            # 尝试方式2：直接读取前几列（假设标准格式）
            if len(df_raw) > 12 and len(df_raw.columns) >= 5:
                data_rows = df_raw.iloc[13:].copy()
                data_df = pd.DataFrame({
                    '日期': pd.to_datetime(data_rows.iloc[:, 4], errors='coerce'),
                    '温度': pd.to_numeric(data_rows.iloc[:, 2], errors='coerce'),
                    '湿度': pd.to_numeric(data_rows.iloc[:, 3], errors='coerce'),
                    '来源文件': file_name
                })
        
        # 去除无效数据（日期、温度、湿度都不能为空）
        data_df = data_df.dropna(subset=['日期', '温度', '湿度'])
        
        print(f"  文件 {file_name}: {len(data_df)} 条有效数据")
        
        return data_df
            
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return pd.DataFrame()


def round_to_15min(dt):
    """
    将时间四舍五入到最近的15分钟间隔（00:00:00, 00:15:00, 00:30:00, 00:45:00等）
    
    Args:
        dt: pandas datetime 列
        
    Returns:
        pandas datetime 列（已四舍五入）
    """
    # 获取分钟数
    minutes = dt.dt.minute
    # 四舍五入：小于7.5分钟舍去，大于等于7.5分钟进位到15分钟间隔
    dt_rounded = dt.dt.floor('h') + pd.to_timedelta((minutes + 7) // 15 * 15, unit='m')
    return dt_rounded


def main():
    """
    主函数：处理所有Excel文件，按时间分组计算平均值，保存结果
    """
    print("开始数据预处理...")
    
    # 定义输入输出目录（scripts目录向上一级为项目根目录）
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir_name = "温湿度数据"  # 读取的文件夹名称
    input_dir = os.path.join(current_dir, input_dir_name)
    output_dir = os.path.join(current_dir, "天气数据", input_dir_name)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 获取所有Excel文件（只处理1.1-1.10）
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在 - {input_dir}")
        return
    
    excel_files = []
    for f in os.listdir(input_dir):
        if f.endswith('.xlsx'):
            # 检查是否为1.1-1.10范围内的文件
            match = os.path.splitext(f)[0]
            if match.startswith('1.'):
                try:
                    num = float(match[2:])
                    if 1 <= num <= 10:
                        excel_files.append(f)
                except:
                    pass
    
    if not excel_files:
        print("未找到符合条件的Excel文件（1.1-1.10）")
        return
    
    # 按文件名排序
    excel_files.sort(key=lambda x: float(os.path.splitext(x)[0][2:]))
    
    print(f"找到 {len(excel_files)} 个符合条件的Excel文件:")
    for f in excel_files:
        print(f"  - {f}")
    
    # 存储所有处理后的数据
    all_data_frames = []
    
    # 处理每个文件
    print("\n正在处理文件...")
    for filename in excel_files:
        file_path = os.path.join(input_dir, filename)
        data_df = process_excel_file(file_path)
        
        if not data_df.empty:
            all_data_frames.append(data_df)
    
    if not all_data_frames:
        print("未处理任何有效数据")
        return
    
    # 合并所有数据
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"\n合并后总数据条数: {len(combined_df)}")
    
    # 判断时间是否全部相同
    unique_times = combined_df['日期'].unique()
    all_times_same = len(unique_times) == 1
    
    if all_times_same:
        print("检测到所有时间相同，不进行四舍五入操作")
        combined_df['分组时间'] = combined_df['日期']
    else:
        print("检测到时间不相同，进行四舍五入到15分钟间隔")
        combined_df['分组时间'] = round_to_15min(combined_df['日期'])
    
    # 按时间分组计算平均值
    print("正在按时间分组计算平均值...")
    grouped = combined_df.groupby('分组时间').agg({
        '温度': 'mean',
        '湿度': 'mean'
    }).reset_index()
    
    # 重命名列
    grouped.columns = ['时间', '温度', '湿度']
    
    # 数值保留两位小数
    grouped['温度'] = grouped['温度'].round(2)
    grouped['湿度'] = grouped['湿度'].round(2)
    
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
    print(f"总体平均温度: {grouped['温度'].mean():.2f}°C")
    print(f"总体平均湿度: {grouped['湿度'].mean():.2f}%")
    
    print("\n数据预处理完成!")


if __name__ == "__main__":
    main()