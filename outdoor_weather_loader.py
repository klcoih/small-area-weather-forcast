#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
户外气象数据采集模块
功能：
  1. 读取宣威市尚营种气象CSV文件（温度、湿度、光照强度、风向、风速、降雨量）
  2. 数据质量检查（时间间隔、异常值、缺失值、特殊字段校验）
  3. 数据标准化（重命名列、添加数据源标记）
  4. 特殊字段处理（降雨量零膨胀、静风风向、夜间光照）
  5. 数据质量报告生成
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


def check_time_interval_outdoor(time_series, tolerance_ratio=0.3):
    """
    检查户外气象数据的时间间隔（允许一定不均匀度）

    Args:
        time_series: pandas datetime Series（已排序）
        tolerance_ratio: 允许的偏差比例，默认0.3（即间隔在期望值的±30%以内视为正常）

    Returns:
        dict: 间隔检查结果
            - 'is_reasonably_uniform': 整体是否大致均匀
            - 'mean_interval': 平均间隔（timedelta）
            - 'median_interval': 中位间隔（timedelta）
            - 'irregular_count': 不规则行数
            - 'irregular_rows': 不规则行索引列表
            - 'interval_stats': 间隔统计（min/max/mean/std）
    """
    if len(time_series) < 2:
        logger.warning("时间序列数据不足两条，无法检查间隔")
        return {
            'is_reasonably_uniform': True,
            'mean_interval': None,
            'median_interval': None,
            'irregular_count': 0,
            'irregular_rows': [],
            'interval_stats': None
        }

    diffs = time_series.diff().dropna()
    median_td = diffs.median()
    mean_td = diffs.mean()

    if pd.isna(median_td):
        logger.warning("无法计算时间间隔中位数")
        return {
            'is_reasonably_uniform': False,
            'mean_interval': mean_td,
            'median_interval': None,
            'irregular_count': len(diffs),
            'irregular_rows': diffs.index.tolist(),
            'interval_stats': None
        }

    median_seconds = median_td.total_seconds()
    if median_seconds == 0:
        logger.warning("时间间隔中位数为0，数据异常")
        return {
            'is_reasonably_uniform': False,
            'mean_interval': mean_td,
            'median_interval': median_td,
            'irregular_count': len(diffs),
            'irregular_rows': diffs.index.tolist(),
            'interval_stats': None
        }

    diffs_seconds = diffs.dt.total_seconds()
    relative_deviation = np.abs(diffs_seconds - median_seconds) / median_seconds
    irregular_mask = relative_deviation > tolerance_ratio
    irregular_indices = diffs[irregular_mask].index.tolist()

    result = {
        'is_reasonably_uniform': len(irregular_indices) <= len(diffs) * 0.1,
        'mean_interval': mean_td,
        'median_interval': median_td,
        'irregular_count': len(irregular_indices),
        'irregular_rows': irregular_indices,
        'interval_stats': {
            'min_seconds': diffs.min().total_seconds(),
            'max_seconds': diffs.max().total_seconds(),
            'mean_seconds': diffs.mean().total_seconds(),
            'std_seconds': diffs.std().total_seconds(),
            'median_seconds': median_seconds
        }
    }

    logger.info(
        f"时间间隔检查：平均 {mean_td}, 中位 {median_td}, "
        f"不规则行 {len(irregular_indices)}/{len(diffs)}"
    )

    return result


def check_anomalies_outdoor(df):
    """
    检测户外气象数据的异常值：
      - 温度: -15 ~ 50 °C
      - 湿度: 0 ~ 100 %
      - 光照强度: >= 0
      - 风向: 0 ~ 360 ° 或 NaN
      - 风速: >= 0
      - 降雨量: >= 0

    Args:
        df: DataFrame

    Returns:
        dict: 各字段异常行索引
    """
    anomaly_report = {}

    temp_anom = df[(df['温度'] < -15.0) | (df['温度'] > 50.0)].index.tolist()
    anomaly_report['温度'] = {'count': len(temp_anom), 'rows': temp_anom}

    hum_anom = df[(df['湿度'] < 0.0) | (df['湿度'] > 100.0)].index.tolist()
    anomaly_report['湿度'] = {'count': len(hum_anom), 'rows': hum_anom}

    light_anom = df[df['光照强度'] < 0.0].index.tolist()
    anomaly_report['光照强度'] = {'count': len(light_anom), 'rows': light_anom}

    wind_dir_anom = df[(df['风向'] < 0.0) | (df['风向'] > 360.0)].index.tolist()
    anomaly_report['风向'] = {'count': len(wind_dir_anom), 'rows': wind_dir_anom}

    wind_spd_anom = df[df['风速'] < 0.0].index.tolist()
    anomaly_report['风速'] = {'count': len(wind_spd_anom), 'rows': wind_spd_anom}

    rain_anom = df[df['降雨量'] < 0.0].index.tolist()
    anomaly_report['降雨量'] = {'count': len(rain_anom), 'rows': rain_anom}

    total = sum(v['count'] for v in anomaly_report.values())
    if total == 0:
        logger.info("异常值检查通过：所有字段均在合理范围内")
    else:
        logger.warning(f"异常值检查：共发现 {total} 处异常")
        for field, info in anomaly_report.items():
            if info['count'] > 0:
                logger.warning(f"  {field}: {info['count']} 条异常")

    return anomaly_report


def handle_missing_values_outdoor(df, method='interpolate'):
    """
    处理户外气象数据的缺失值

    Args:
        df: DataFrame
        method: 'interpolate' / 'drop' / 'ffill' / 'bfill'

    Returns:
        DataFrame: 处理后的数据
    """
    numeric_cols = ['温度', '湿度', '光照强度', '风向', '风速', '降雨量']

    missing_report = {}
    for col in numeric_cols:
        cnt = df[col].isnull().sum()
        if cnt > 0:
            missing_report[col] = cnt

    if not missing_report:
        logger.info("缺失值检查通过：数据完整无缺失")
        return df.copy()

    logger.warning(f"缺失值检查：共 {sum(missing_report.values())} 个缺失值")
    for col, cnt in missing_report.items():
        logger.warning(f"  {col}: {cnt} 个缺失")

    df_clean = df.copy()

    if method == 'drop':
        df_clean = df_clean.dropna(subset=numeric_cols)
        logger.info(f"删除含缺失值行后剩余 {len(df_clean)} 条")
    elif method == 'interpolate':
        for col in numeric_cols:
            if col in ['风向']:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
            else:
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        logger.info("已通过插值填充缺失值（风向使用前后填充）")
    elif method == 'ffill':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(method='ffill')
        logger.info("已通过前向填充处理缺失值")
    elif method == 'bfill':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(method='bfill')
        logger.info("已通过后向填充处理缺失值")

    return df_clean


def check_special_fields(df):
    """
    特殊字段检查：
      - 降雨量：零膨胀检查（记录零值比例）
      - 风向：静风检查（风速≈0时风向无实际意义）
      - 光照强度：夜间检查（夜间应为0或接近0）

    Args:
        df: DataFrame

    Returns:
        dict: 特殊字段检查报告
    """
    report = {}

    rainfall_zero_ratio = (df['降雨量'] == 0).mean()
    report['降雨量_零值比例'] = round(rainfall_zero_ratio * 100, 2)
    report['降雨量_零值条数'] = int((df['降雨量'] == 0).sum())
    report['降雨量_非零条数'] = int((df['降雨量'] > 0).sum())
    if df['降雨量'].max() > 0:
        report['降雨量_最大值'] = round(df['降雨量'].max(), 2)
    logger.info(
        f"降雨量零膨胀检查：{report['降雨量_零值比例']}% 为零值 "
        f"({report['降雨量_零值条数']}/{len(df)})"
    )

    calm_mask = df['风速'] == 0
    calm_count = int(calm_mask.sum())
    report['静风记录数'] = calm_count
    report['静风比例'] = round(calm_count / len(df) * 100, 2)
    logger.info(
        f"静风检查：{calm_count} 条静风记录（风速=0），占比 {report['静风比例']}%，风向无实际意义"
    )

    df_with_hour = df.copy()
    df_with_hour['hour'] = df['时间'].dt.hour
    night_mask = (df_with_hour['hour'] < 6) | (df_with_hour['hour'] >= 20)
    night_data = df_with_hour[night_mask]

    night_total = len(night_data)
    if night_total > 0:
        night_light_zero = int((night_data['光照强度'] == 0).sum())
        night_light_nonzero = night_total - night_light_zero
        report['夜间记录数'] = night_total
        report['夜间光照为零数'] = night_light_zero
        report['夜间光照非零数'] = night_light_nonzero
        if night_light_nonzero > 0:
            report['夜间光照非零最大值'] = round(night_data['光照强度'].max(), 2)
        logger.info(
            f"夜间光照检查（20:00-06:00）：共 {night_total} 条，"
            f"光照为0的 {night_light_zero} 条，非0的 {night_light_nonzero} 条"
        )
    else:
        report['夜间记录数'] = 0
        report['夜间光照为零数'] = 0
        report['夜间光照非零数'] = 0
        logger.info("夜间光照检查：无夜间数据")

    return report


def standardize_data_outdoor(df):
    """
    户外气象数据标准化：
      - 重命名列：温度→temperature_outdoor，湿度→humidity_outdoor，
                  光照强度→light_intensity，风向→wind_direction，
                  风速→wind_speed，降雨量→rainfall，时间→timestamp
      - 添加数据源标记：location='outdoor'

    Args:
        df: 原始DataFrame

    Returns:
        DataFrame: 标准化后的数据
    """
    df_std = df.copy()

    column_mapping = {
        '时间': 'timestamp',
        '温度': 'temperature_outdoor',
        '湿度': 'humidity_outdoor',
        '光照强度': 'light_intensity',
        '风向': 'wind_direction',
        '风速': 'wind_speed',
        '降雨量': 'rainfall'
    }
    existing_mapping = {k: v for k, v in column_mapping.items() if k in df_std.columns}
    df_std = df_std.rename(columns=existing_mapping)

    df_std['location'] = 'outdoor'

    logger.info(f"数据标准化完成：重命名列 {list(existing_mapping.values())}，添加 location='outdoor'")

    return df_std


class OutdoorWeatherLoader:
    """
    户外气象数据加载器

    支持功能：
      - 从CSV文件加载多字段气象数据
      - 时间间隔不均匀检查与处理
      - 多字段异常值检测
      - 特殊字段校验（降雨零膨胀、静风风向、夜间光照）
      - 数据标准化与质量报告生成

    使用示例:
        loader = OutdoorWeatherLoader('天气数据/宣威市尚营种气象/宣威市尚营种气象.csv')
        loader.load()
        quality_report = loader.check_quality()
        df_std = loader.standardize()
        report = loader.generate_report()
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
        从CSV文件加载户外气象数据

        Returns:
            DataFrame: 原始数据
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")

        logger.info(f"正在加载数据文件: {self.csv_path}")

        self.df_raw = pd.read_csv(self.csv_path, encoding='utf-8')

        if '时间' in self.df_raw.columns:
            self.df_raw['时间'] = pd.to_datetime(self.df_raw['时间'], errors='coerce')

        numeric_cols = ['温度', '湿度', '光照强度', '风向', '风速', '降雨量']
        for col in numeric_cols:
            if col in self.df_raw.columns:
                self.df_raw[col] = pd.to_numeric(self.df_raw[col], errors='coerce')

        if '序号' in self.df_raw.columns:
            self.df_raw = self.df_raw.drop(columns=['序号'])

        self.df_raw = self.df_raw.dropna(subset=['时间'])
        self.df_raw = self.df_raw.sort_values('时间').reset_index(drop=True)

        logger.info(
            f"数据加载成功：共 {len(self.df_raw)} 条记录，"
            f"时间范围 {self.df_raw['时间'].min()} 至 {self.df_raw['时间'].max()}"
        )

        return self.df_raw

    def check_quality(self, tolerance_ratio=0.3):
        """
        执行完整的数据质量检查

        Args:
            tolerance_ratio: 时间间隔允许偏差比例

        Returns:
            dict: 质量检查报告
        """
        if self.df_raw is None:
            raise ValueError("请先调用 load() 方法加载数据")

        logger.info("=" * 60)
        logger.info("开始户外气象数据质量检查...")
        logger.info("=" * 60)

        interval_result = check_time_interval_outdoor(self.df_raw['时间'], tolerance_ratio)

        anomaly_result = check_anomalies_outdoor(self.df_raw)

        self.df_clean = handle_missing_values_outdoor(self.df_raw, method='interpolate')

        special_report = check_special_fields(self.df_clean)

        self.quality_report = {
            'total_records': len(self.df_raw),
            'time_range': {
                'start': self.df_raw['时间'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': self.df_raw['时间'].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            'time_interval_check': interval_result,
            'anomaly_check': anomaly_result,
            'special_fields_check': special_report,
            'cleaned_records': len(self.df_clean),
            'data_loss': len(self.df_raw) - len(self.df_clean)
        }

        logger.info("=" * 60)
        logger.info("数据质量检查完成")
        logger.info(f"  总记录数: {self.quality_report['total_records']}")
        logger.info(f"  清洗后记录数: {self.quality_report['cleaned_records']}")
        logger.info(f"  降雨量零值比例: {special_report['降雨量_零值比例']}%")
        logger.info(f"  静风记录: {special_report['静风记录数']} 条")
        logger.info("=" * 60)

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

        self.df_standardized = standardize_data_outdoor(self.df_clean)

        std_cols = [c for c in ['timestamp', 'temperature_outdoor', 'humidity_outdoor',
                                 'light_intensity', 'wind_direction', 'wind_speed',
                                 'rainfall', 'location'] if c in self.df_standardized.columns]
        display_df = self.df_standardized[std_cols].copy()
        for col in display_df.columns:
            if col != 'timestamp' and col != 'location' and display_df[col].dtype in ['float64', 'int64']:
                display_df[col] = display_df[col].round(2)

        logger.info(f"标准化数据共 {len(self.df_standardized)} 条")
        if 'temperature_outdoor' in display_df.columns:
            logger.info(
                f"  温度范围: {display_df['temperature_outdoor'].min():.2f} ~ "
                f"{display_df['temperature_outdoor'].max():.2f} °C"
            )
        if 'humidity_outdoor' in display_df.columns:
            logger.info(
                f"  湿度范围: {display_df['humidity_outdoor'].min():.2f} ~ "
                f"{display_df['humidity_outdoor'].max():.2f} %"
            )

        return self.df_standardized

    def load_batch(self, start_time=None, end_time=None):
        """
        批量加载指定时间范围的历史数据

        Args:
            start_time: 起始时间（字符串或datetime）
            end_time: 结束时间（字符串或datetime）

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
            f"批量加载数据：{len(df_filtered)} 条，"
            f"时间 {df_filtered['timestamp'].min()} 至 {df_filtered['timestamp'].max()}"
        )

        return df_filtered.reset_index(drop=True)

    def stream_realtime(self, interval=1.0, start_index=0, end_index=None):
        """
        生成器：逐条读取数据，模拟实时气象数据流入

        Args:
            interval: 输出间隔（秒）
            start_index: 起始索引
            end_index: 结束索引

        Yields:
            dict: 单条记录
        """
        if self.df_standardized is None:
            self.standardize()

        if end_index is None:
            end_index = len(self.df_standardized)

        total = end_index - start_index
        logger.info(f"开始实时数据模拟，共 {total} 条，间隔 {interval} 秒/条")

        for i in range(start_index, min(end_index, len(self.df_standardized))):
            row = self.df_standardized.iloc[i]
            record = {
                'index': i,
                'timestamp': row['timestamp'],
                'temperature_outdoor': round(row.get('temperature_outdoor', np.nan), 2),
                'humidity_outdoor': round(row.get('humidity_outdoor', np.nan), 2),
                'light_intensity': round(row.get('light_intensity', np.nan), 2),
                'wind_direction': round(row.get('wind_direction', np.nan), 2),
                'wind_speed': round(row.get('wind_speed', np.nan), 2),
                'rainfall': round(row.get('rainfall', np.nan), 2),
                'location': row.get('location', 'outdoor')
            }
            yield record
            time.sleep(interval)

        logger.info("实时数据模拟完成")

    def generate_report(self):
        """
        生成完整的数据质量报告（可读文本格式）

        Returns:
            str: 格式化的质量报告
        """
        if self.quality_report is None:
            raise ValueError("请先调用 check_quality() 方法")

        qr = self.quality_report
        lines = []
        lines.append("=" * 60)
        lines.append("  户外气象数据质量报告")
        lines.append("=" * 60)
        lines.append(f"  数据文件: {self.csv_path}")
        lines.append(f"  总记录数: {qr['total_records']}")
        lines.append(f"  时间范围: {qr['time_range']['start']} ~ {qr['time_range']['end']}")
        lines.append(f"  清洗后记录: {qr['cleaned_records']} (丢失 {qr['data_loss']} 条)")

        ti = qr['time_interval_check']
        lines.append("-" * 40)
        lines.append("  [时间间隔检查]")
        if ti['interval_stats']:
            s = ti['interval_stats']
            lines.append(f"    均值: {s['mean_seconds']:.1f}s  中位: {s['median_seconds']:.1f}s")
            lines.append(f"    最小: {s['min_seconds']:.1f}s  最大: {s['max_seconds']:.1f}s")
        lines.append(f"    不规则行数: {ti['irregular_count']}")
        lines.append(f"    大致均匀: {ti['is_reasonably_uniform']}")

        ac = qr['anomaly_check']
        lines.append("-" * 40)
        lines.append("  [异常值检查]")
        total_anom = sum(v['count'] for v in ac.values())
        for field, info in ac.items():
            lines.append(f"    {field}: {info['count']} 条异常")
        lines.append(f"    总异常数: {total_anom}")

        sf = qr['special_fields_check']
        lines.append("-" * 40)
        lines.append("  [特殊字段检查]")
        lines.append(f"    降雨量零值比例: {sf['降雨量_零值比例']}% ({sf['降雨量_零值条数']}/{qr['total_records']})")
        if '降雨量_最大值' in sf:
            lines.append(f"    降雨量最大值: {sf['降雨量_最大值']} mm")
        lines.append(f"    静风记录: {sf['静风记录数']} 条 ({sf['静风比例']}%)")
        if sf.get('夜间记录数', 0) > 0:
            lines.append(f"    夜间记录 (20:00-06:00): {sf['夜间记录数']} 条")
            lines.append(f"      光照为0: {sf['夜间光照为零数']} 条")
            lines.append(f"      光照非0: {sf.get('夜间光照非零数', 0)} 条")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_statistics(self):
        """
        获取各字段统计摘要

        Returns:
            dict: 统计信息
        """
        if self.df_standardized is None:
            self.standardize()

        df = self.df_standardized

        def field_stats(col):
            if col not in df.columns:
                return None
            return {
                'mean': round(df[col].mean(), 2),
                'min': round(df[col].min(), 2),
                'max': round(df[col].max(), 2),
                'std': round(df[col].std(), 2)
            }

        stats = {
            'total_records': len(df),
            'time_range': {
                'start': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            'temperature_outdoor': field_stats('temperature_outdoor'),
            'humidity_outdoor': field_stats('humidity_outdoor'),
            'light_intensity': field_stats('light_intensity'),
            'wind_direction': field_stats('wind_direction'),
            'wind_speed': field_stats('wind_speed'),
            'rainfall': field_stats('rainfall')
        }

        return stats


def main():
    """
    使用示例：演示 OutdoorWeatherLoader 的完整功能
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '天气数据', '宣威市尚营种气象', '宣威市尚营种气象.csv')

    print("=" * 60)
    print("  户外气象数据采集模块 - 功能演示")
    print("=" * 60)

    loader = OutdoorWeatherLoader(csv_path)

    print("\n[1] 加载数据")
    print("-" * 40)
    loader.load()

    print("\n[2] 数据质量检查")
    print("-" * 40)
    loader.check_quality()

    print("\n[3] 生成质量报告")
    print("-" * 40)
    report = loader.generate_report()
    print(report)

    print("\n[4] 数据标准化")
    print("-" * 40)
    df_std = loader.standardize()
    std_cols = [c for c in ['timestamp', 'temperature_outdoor', 'humidity_outdoor',
                             'light_intensity', 'wind_direction', 'wind_speed',
                             'rainfall', 'location'] if c in df_std.columns]
    print("\n标准化后数据预览（前5条）:")
    print(df_std[std_cols].head().to_string(index=False))

    print("\n[5] 统计摘要")
    print("-" * 40)
    stats = loader.get_statistics()
    print(f"  总记录数: {stats['total_records']}")
    for key, val in stats.items():
        if isinstance(val, dict) and 'mean' in val:
            print(f"  {key}: 均值 {val['mean']}, 范围 [{val['min']}, {val['max']}], 标准差 {val['std']}")

    print("\n[6] 实时数据模拟（前3条）")
    print("-" * 40)
    for i, record in enumerate(loader.stream_realtime(interval=0.1, end_index=3)):
        print(f"  [{i+1}] {record['timestamp']} | "
              f"温度: {record['temperature_outdoor']}°C | "
              f"湿度: {record['humidity_outdoor']}% | "
              f"风速: {record['wind_speed']}m/s")

    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()