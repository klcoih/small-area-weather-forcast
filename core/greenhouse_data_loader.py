#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
温室大棚温湿度数据采集模块
功能：
  1. 读取温湿度CSV文件
  2. 数据质量检查（时间间隔、异常值、缺失值）
  3. 数据标准化（重命名列、添加数据源标记）
  4. 实时数据模拟（逐条读取）和批量历史数据加载
"""

import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_time_interval(time_series, expected_interval_minutes=15.0):
    """
    检查时间序列的间隔是否均匀

    Args:
        time_series: pandas datetime Series（已排序）
        expected_interval_minutes: 期望的时间间隔（分钟），默认15分钟

    Returns:
        dict: 包含检查结果的字典
            - 'is_uniform': 是否均匀
            - 'expected_interval': 期望间隔（timedelta）
            - 'irregular_rows': 不均匀行的索引列表
            - 'actual_intervals': 实际间隔的统计信息
    """
    if len(time_series) < 2:
        logger.warning("时间序列数据不足两条，无法检查间隔")
        return {
            'is_uniform': True,
            'expected_interval': timedelta(minutes=expected_interval_minutes),
            'irregular_rows': [],
            'actual_intervals': None
        }

    expected_td = timedelta(minutes=expected_interval_minutes)
    diffs = time_series.diff().dropna()
    irregular_mask = diffs != expected_td
    irregular_indices = diffs[irregular_mask].index.tolist()

    result = {
        'is_uniform': len(irregular_indices) == 0,
        'expected_interval': expected_td,
        'irregular_rows': irregular_indices,
        'actual_intervals': {
            'min': diffs.min(),
            'max': diffs.max(),
            'mean': diffs.mean(),
            'std': diffs.std()
        }
    }

    if result['is_uniform']:
        logger.info(f"时间间隔检查通过：所有间隔均为 {expected_interval_minutes} 分钟")
    else:
        logger.warning(
            f"时间间隔不均匀：发现 {len(irregular_indices)} 处不规则间隔，"
            f"范围 [{diffs.min()}, {diffs.max()}]"
        )

    return result


def check_anomalies(df, temp_col='温度', humidity_col='湿度',
                    temp_range=(-10.0, 50.0), humidity_range=(0.0, 100.0)):
    """
    检测温度和湿度的异常值

    Args:
        df: DataFrame，包含温度和湿度列
        temp_col: 温度列名
        humidity_col: 湿度列名
        temp_range: 温度合理范围 (min, max)
        humidity_range: 湿度合理范围 (min, max)

    Returns:
        dict: 异常值检测结果
            - 'temperature_anomalies': 温度异常的行索引列表
            - 'humidity_anomalies': 湿度异常的行索引列表
            - 'total_anomalies': 总异常数
    """
    temp_min, temp_max = temp_range
    hum_min, hum_max = humidity_range

    temp_anomalies = df[
        (df[temp_col] < temp_min) | (df[temp_col] > temp_max)
    ].index.tolist()

    humidity_anomalies = df[
        (df[humidity_col] < hum_min) | (df[humidity_col] > hum_max)
    ].index.tolist()

    result = {
        'temperature_anomalies': temp_anomalies,
        'humidity_anomalies': humidity_anomalies,
        'total_anomalies': len(temp_anomalies) + len(humidity_anomalies)
    }

    if result['total_anomalies'] == 0:
        logger.info("异常值检查通过：未发现超出合理范围的数据")
    else:
        logger.warning(
            f"异常值检查：发现 {len(temp_anomalies)} 条温度异常、"
            f"{len(humidity_anomalies)} 条湿度异常"
        )

    return result


def handle_missing_values(df, columns=None, method='interpolate'):
    """
    处理缺失值

    Args:
        df: DataFrame
        columns: 需要处理的列名列表，默认为 ['温度', '湿度']
        method: 处理方法
            - 'interpolate': 线性插值（默认）
            - 'drop': 删除含缺失值的行
            - 'ffill': 前向填充
            - 'bfill': 后向填充

    Returns:
        DataFrame: 处理后的数据
    """
    if columns is None:
        columns = ['温度', '湿度']

    missing_count = df[columns].isnull().sum().sum()
    if missing_count == 0:
        logger.info("缺失值检查通过：数据完整无缺失")
        return df.copy()

    logger.warning(f"发现 {missing_count} 个缺失值，使用 {method} 方法处理")

    df_clean = df.copy()

    if method == 'drop':
        df_clean = df_clean.dropna(subset=columns)
        logger.info(f"删除缺失值后剩余 {len(df_clean)} 条数据")
    elif method == 'interpolate':
        for col in columns:
            df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        logger.info("已通过线性插值填充缺失值")
    elif method == 'ffill':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(method='ffill')
        logger.info("已通过前向填充处理缺失值")
    elif method == 'bfill':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(method='bfill')
        logger.info("已通过后向填充处理缺失值")

    return df_clean


def standardize_data(df):
    """
    数据标准化：
      - 重命名列：温度→temperature，湿度→humidity，时间→timestamp
      - 添加数据源标记：location='greenhouse'

    Args:
        df: 原始DataFrame（含 '时间'、'温度'、'湿度' 列）

    Returns:
        DataFrame: 标准化后的数据
    """
    df_std = df.copy()

    column_mapping = {
        '时间': 'timestamp',
        '温度': 'temperature',
        '湿度': 'humidity'
    }
    existing_mapping = {k: v for k, v in column_mapping.items() if k in df_std.columns}
    df_std = df_std.rename(columns=existing_mapping)

    df_std['location'] = 'greenhouse'

    logger.info(f"数据标准化完成：重命名列 {list(existing_mapping.values())}，添加 location='greenhouse'")

    return df_std


class GreenhouseDataLoader:
    """
    温室大棚温湿度数据加载器

    支持功能：
      - 从CSV文件加载温湿度数据
      - 按时间顺序逐条读取，模拟实时数据流入
      - 批量历史数据加载
      - 数据质量检查与标准化一站式处理

    使用示例:
        loader = GreenhouseDataLoader('天气数据/温湿度数据/温湿度数据.csv')
        # 如果从项目根运行:
        # loader = GreenhouseDataLoader('../天气数据/温湿度数据/温湿度数据.csv')
        loader.load()

        # 质量检查
        loader.check_quality()

        # 标准化
        df_std = loader.standardize()

        # 实时模拟
        for record in loader.stream_realtime(interval=1.0):
            print(record)
    """

    def __init__(self, csv_path):
        """
        初始化数据加载器

        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
        self.df_raw = None
        self.df_clean = None
        self.df_standardized = None
        self.quality_report = None

    def load(self):
        """
        从CSV文件加载数据

        Returns:
            DataFrame: 原始数据
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")

        logger.info(f"正在加载数据文件: {self.csv_path}")

        self.df_raw = pd.read_csv(self.csv_path, encoding='utf-8')

        if '时间' in self.df_raw.columns:
            self.df_raw['时间'] = pd.to_datetime(self.df_raw['时间'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        if '温度' in self.df_raw.columns:
            self.df_raw['温度'] = pd.to_numeric(self.df_raw['温度'], errors='coerce')

        if '湿度' in self.df_raw.columns:
            self.df_raw['湿度'] = pd.to_numeric(self.df_raw['湿度'], errors='coerce')

        if '序号' in self.df_raw.columns:
            self.df_raw = self.df_raw.drop(columns=['序号'])

        self.df_raw = self.df_raw.sort_values('时间').reset_index(drop=True)

        logger.info(
            f"数据加载成功：共 {len(self.df_raw)} 条记录，"
            f"时间范围 {self.df_raw['时间'].min()} 至 {self.df_raw['时间'].max()}"
        )

        return self.df_raw

    def check_quality(self, expected_interval_minutes=15.0,
                      temp_range=(-10.0, 50.0), humidity_range=(0.0, 100.0)):
        """
        执行完整的数据质量检查

        Args:
            expected_interval_minutes: 期望时间间隔（分钟）
            temp_range: 温度合理范围
            humidity_range: 湿度合理范围

        Returns:
            dict: 质量检查报告
        """
        if self.df_raw is None:
            raise ValueError("请先调用 load() 方法加载数据")

        logger.info("=" * 50)
        logger.info("开始数据质量检查...")
        logger.info("=" * 50)

        interval_result = check_time_interval(
            self.df_raw['时间'], expected_interval_minutes
        )

        anomaly_result = check_anomalies(
            self.df_raw, temp_range=temp_range, humidity_range=humidity_range
        )

        self.df_clean = handle_missing_values(self.df_raw, method='interpolate')

        self.quality_report = {
            'total_records': len(self.df_raw),
            'time_interval_check': interval_result,
            'anomaly_check': anomaly_result,
            'cleaned_records': len(self.df_clean),
            'data_loss': len(self.df_raw) - len(self.df_clean)
        }

        logger.info("=" * 50)
        logger.info("数据质量检查完成")
        logger.info(
            f"  总记录数: {self.quality_report['total_records']}"
        )
        logger.info(
            f"  时间间隔均匀: {interval_result['is_uniform']}"
        )
        logger.info(
            f"  异常值数量: {anomaly_result['total_anomalies']}"
        )
        logger.info("=" * 50)

        return self.quality_report

    def standardize(self):
        """
        对清洗后的数据进行标准化

        Returns:
            DataFrame: 标准化后的数据
        """
        if self.df_clean is None:
            if self.df_raw is None:
                raise ValueError("请先调用 load() 和 check_quality() 方法")
            self.df_clean = self.df_raw.copy()

        self.df_standardized = standardize_data(self.df_clean)

        display_df = self.df_standardized[['timestamp', 'temperature', 'humidity', 'location']].copy()
        display_df['temperature'] = display_df['temperature'].round(2)
        display_df['humidity'] = display_df['humidity'].round(2)

        logger.info(f"标准化数据共 {len(self.df_standardized)} 条")
        logger.info(
            f"  温度范围: {display_df['temperature'].min():.2f} ~ {display_df['temperature'].max():.2f} °C"
        )
        logger.info(
            f"  湿度范围: {display_df['humidity'].min():.2f} ~ {display_df['humidity'].max():.2f} %"
        )

        return self.df_standardized

    def load_batch(self, start_time=None, end_time=None):
        """
        批量加载指定时间范围的历史数据

        Args:
            start_time: 起始时间（字符串或datetime），默认全部
            end_time: 结束时间（字符串或datetime），默认全部

        Returns:
            DataFrame: 指定时间范围内的标准化数据
        """
        if self.df_standardized is None:
            self.standardize()

        df_filtered = self.df_standardized.copy()

        if start_time is not None:
            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time)
            df_filtered = df_filtered[df_filtered['timestamp'] >= start_time]

        if end_time is not None:
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time)
            df_filtered = df_filtered[df_filtered['timestamp'] <= end_time]

        logger.info(
            f"批量加载数据：{len(df_filtered)} 条记录，"
            f"时间范围 {df_filtered['timestamp'].min()} 至 {df_filtered['timestamp'].max()}"
        )

        return df_filtered.reset_index(drop=True)

    def stream_realtime(self, interval=1.0, start_index=0, end_index=None):
        """
        生成器：按时间顺序逐条读取数据，模拟实时数据流入

        Args:
            interval: 每条数据的输出间隔（秒），默认1.0秒
            start_index: 起始索引
            end_index: 结束索引（不包含），默认全部

        Yields:
            dict: 单条记录 {timestamp, temperature, humidity, location}
        """
        if self.df_standardized is None:
            self.standardize()

        if end_index is None:
            end_index = len(self.df_standardized)

        total = end_index - start_index
        logger.info(f"开始实时数据模拟，共 {total} 条记录，间隔 {interval} 秒/条")

        for i in range(start_index, min(end_index, len(self.df_standardized))):
            row = self.df_standardized.iloc[i]
            record = {
                'index': i,
                'timestamp': row['timestamp'],
                'temperature': round(row['temperature'], 2),
                'humidity': round(row['humidity'], 2),
                'location': row['location']
            }
            yield record
            time.sleep(interval)

        logger.info("实时数据模拟完成")

    def get_statistics(self):
        """
        获取数据统计摘要

        Returns:
            dict: 统计信息
        """
        if self.df_standardized is None:
            self.standardize()

        df = self.df_standardized
        stats = {
            'total_records': len(df),
            'time_range': {
                'start': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            'temperature': {
                'mean': round(df['temperature'].mean(), 2),
                'min': round(df['temperature'].min(), 2),
                'max': round(df['temperature'].max(), 2),
                'std': round(df['temperature'].std(), 2)
            },
            'humidity': {
                'mean': round(df['humidity'].mean(), 2),
                'min': round(df['humidity'].min(), 2),
                'max': round(df['humidity'].max(), 2),
                'std': round(df['humidity'].std(), 2)
            }
        }

        return stats


def main():
    """
    使用示例：演示 GreenhouseDataLoader 的完整功能
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '天气数据', '温湿度数据', '温湿度数据.csv')

    print("=" * 60)
    print("  温室大棚温湿度数据采集模块 - 功能演示")
    print("=" * 60)

    loader = GreenhouseDataLoader(csv_path)

    print("\n[1] 加载数据")
    print("-" * 40)
    loader.load()

    print("\n[2] 数据质量检查")
    print("-" * 40)
    quality_report = loader.check_quality()

    print("\n[3] 数据标准化")
    print("-" * 40)
    df_std = loader.standardize()
    print("\n标准化后数据预览（前5条）:")
    print(df_std[['timestamp', 'temperature', 'humidity', 'location']].head().to_string(index=False))

    print("\n[4] 统计摘要")
    print("-" * 40)
    stats = loader.get_statistics()
    print(f"  总记录数: {stats['total_records']}")
    print(f"  时间范围: {stats['time_range']['start']} ~ {stats['time_range']['end']}")
    print(f"  温度: 均值 {stats['temperature']['mean']}°C, "
          f"范围 [{stats['temperature']['min']}, {stats['temperature']['max']}]°C, "
          f"标准差 {stats['temperature']['std']}")
    print(f"  湿度: 均值 {stats['humidity']['mean']}%, "
          f"范围 [{stats['humidity']['min']}, {stats['humidity']['max']}]%, "
          f"标准差 {stats['humidity']['std']}")

    print("\n[5] 批量历史数据加载（示例：加载前10条）")
    print("-" * 40)
    batch_df = loader.load_batch()
    print(batch_df[['timestamp', 'temperature', 'humidity']].head(10).to_string(index=False))

    print("\n[6] 实时数据模拟（示例：流式输出前5条）")
    print("-" * 40)
    for i, record in enumerate(loader.stream_realtime(interval=0.1, end_index=5)):
        print(f"  [{i+1}] {record['timestamp']} | "
              f"温度: {record['temperature']}°C | "
              f"湿度: {record['humidity']}%")

    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()