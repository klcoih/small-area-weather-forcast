#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块 —— 15分钟间隔环境数据

功能：
  1. 时间对齐：不规则时间→15分钟整点，缺失填充，重复取平均
  2. 数据清洗：异常值检测（3σ/IQR），缺失值处理，可选平滑
  3. 特征工程：滞后特征、时间特征、滚动统计特征
  4. 特殊预处理器：风向→sin/cos，降雨→二分类，夜间光照标记
  5. 多目标关联特征：温差、湿度差、温湿度交互
  6. 数据导出：按目标独立导出 X/y，时间序列划分训练/验证/测试集

使用示例:
    from data_preprocessing import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    df_greenhouse, df_outdoor = pipeline.load_data(...)
    df_merged = pipeline.merge_sources(df_greenhouse, df_outdoor)
    df_aligned = pipeline.align_time(df_merged)
    df_clean = pipeline.clean(df_aligned)
    df_features = pipeline.engineer_features(df_clean)
    pipeline.export_for_training(df_features, target="temperature")
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. 时间对齐类
# ═══════════════════════════════════════════════════════════════

class TimeAligner:
    """
    将不规则时间对齐到15分钟整点

    功能：
      - 对齐到 00, 15, 30, 45 分
      - 缺失时间点用插值填充
      - 重复时间点取平均
    """

    def __init__(self, freq_minutes: int = 15):
        self.freq_minutes = freq_minutes
        self.freq_str = f"{freq_minutes}min"

    def round_to_grid(self, dt: pd.Timestamp) -> pd.Timestamp:
        """将时间对齐到最近的15分钟整点"""
        minute = dt.minute
        aligned_minute = (minute // self.freq_minutes) * self.freq_minutes
        return dt.replace(minute=aligned_minute, second=0, microsecond=0)

    def align(self, df: pd.DataFrame, time_col: str = 'timestamp',
              fill_method: str = 'linear',
              duplicate_method: str = 'mean') -> pd.DataFrame:
        """
        执行时间对齐

        Args:
            df:              含时间列的数据
            time_col:        时间列名
            fill_method:     缺失填充方式 (linear / ffill / bfill)
            duplicate_method: 重复处理方式 (mean / first / last)

        Returns:
            DataFrame: 对齐后的数据
        """
        df = df.copy()
        original_len = len(df)

        df[time_col] = pd.to_datetime(df[time_col])
        df[time_col] = df[time_col].apply(self.round_to_grid)

        dup_count = df.duplicated(subset=[time_col]).sum()
        if dup_count > 0:
            logger.info(f"发现 {dup_count} 条重复时间点，使用 {duplicate_method} 处理")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_dict = {col: duplicate_method for col in numeric_cols if col != time_col}
            non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != time_col]
            for col in non_numeric_cols:
                agg_dict[col] = 'first'
            df = df.groupby(time_col, as_index=False).agg(agg_dict)

        df = df.sort_values(time_col).reset_index(drop=True)

        start = df[time_col].min()
        end = df[time_col].max()
        full_index = pd.date_range(start=start, end=end, freq=self.freq_str)
        full_df = pd.DataFrame({time_col: full_index})

        df = full_df.merge(df, on=time_col, how='left')

        missing_count = df.drop(columns=[time_col]).isnull().any(axis=1).sum()
        if missing_count > 0:
            logger.info(f"缺失 {missing_count} 个时间点，使用 {fill_method} 填充")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if fill_method == 'linear':
                for col in numeric_cols:
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                non_numeric = [c for c in df.columns if c not in numeric_cols and c != time_col]
                for col in non_numeric:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            elif fill_method == 'ffill':
                df = df.fillna(method='ffill')
            elif fill_method == 'bfill':
                df = df.fillna(method='bfill')

        logger.info(
            f"时间对齐完成: {original_len} → {len(df)} 条, "
            f"时间 {df[time_col].min()} ~ {df[time_col].max()}"
        )
        return df


# ═══════════════════════════════════════════════════════════════
# 2. 数据清洗
# ═══════════════════════════════════════════════════════════════

class DataCleaner:
    """
    数据清洗：异常值检测、缺失值处理、可选平滑
    """

    def __init__(self, iqr_factor: float = 1.5, sigma_factor: float = 3.0):
        self.iqr_factor = iqr_factor
        self.sigma_factor = sigma_factor

    def detect_anomalies_iqr(self, series: pd.Series) -> pd.Series:
        """IQR 方法检测异常值，返回布尔掩码"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_factor * iqr
        upper = q3 + self.iqr_factor * iqr
        return (series < lower) | (series > upper)

    def detect_anomalies_sigma(self, series: pd.Series) -> pd.Series:
        """3σ 方法检测异常值，返回布尔掩码"""
        mean = series.mean()
        std = series.std()
        return np.abs(series - mean) > self.sigma_factor * std

    def clean(self, df: pd.DataFrame, time_col: str = 'timestamp',
              anomaly_method: str = 'iqr',
              missing_method: str = 'linear',
              smoothing: bool = False,
              smoothing_window: int = 3,
              columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        执行完整数据清洗

        Args:
            df:              输入数据
            time_col:        时间列名
            anomaly_method:  iqr / sigma
            missing_method:  linear / ffill / drop
            smoothing:       是否平滑
            smoothing_window: 平滑窗口
            columns:         需要清洗的列，默认所有数值列

        Returns:
            (清洗后DataFrame, 清洗报告)
        """
        df_clean = df.copy()
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

        report = {'total': len(df_clean), 'anomalies': {}, 'filled_missing': {}}

        for col in columns:
            if col not in df_clean.columns:
                continue

            series = df_clean[col].copy()

            nan_mask = series.isnull()
            report['filled_missing'][col] = int(nan_mask.sum())

            valid = series.dropna()
            if len(valid) < 4:
                continue

            if anomaly_method == 'iqr':
                anomaly_mask = self.detect_anomalies_iqr(series)
            else:
                anomaly_mask = self.detect_anomalies_sigma(series)

            anomaly_count = int(anomaly_mask.sum())
            report['anomalies'][col] = anomaly_count
            if anomaly_count > 0:
                logger.info(f"  {col}: {anomaly_count} 个异常值 ({anomaly_method}方法)")

            series[anomaly_mask] = np.nan

            if series.isnull().sum() > 0:
                if missing_method == 'linear':
                    series = series.interpolate(method='linear', limit_direction='both')
                elif missing_method == 'ffill':
                    series = series.fillna(method='ffill').fillna(method='bfill')

            if smoothing and len(series) >= smoothing_window:
                series = series.rolling(window=smoothing_window, center=True,
                                        min_periods=1).mean()

            df_clean[col] = series

        total_anomalies = sum(report['anomalies'].values())
        total_filled = sum(report['filled_missing'].values())
        logger.info(
            f"数据清洗完成: {total_anomalies} 个异常处理, {total_filled} 个缺失填充"
        )
        return df_clean, report


# ═══════════════════════════════════════════════════════════════
# 3. 特殊预处理器
# ═══════════════════════════════════════════════════════════════

class WindDirectionPreprocessor:
    """风向预处理器：角度 → sin/cos 向量转换"""

    @staticmethod
    def transform(df: pd.DataFrame, wind_col: str = 'wind_direction',
                  drop_original: bool = False) -> pd.DataFrame:
        """
        将风向角度转换为 sin 和 cos 分量

        Args:
            df:         DataFrame
            wind_col:   风向列名（度，0-360）
            drop_original: 是否删除原始列

        Returns:
            DataFrame: 添加了 wind_sin, wind_cos 列
        """
        if wind_col not in df.columns:
            logger.warning(f"风向列 '{wind_col}' 不存在，跳过转换")
            return df

        df = df.copy()
        radians = np.radians(df[wind_col].fillna(0))
        df['wind_sin'] = np.sin(radians).round(6)
        df['wind_cos'] = np.cos(radians).round(6)

        if drop_original:
            df = df.drop(columns=[wind_col])

        logger.info("风向转换完成: 角度 → (sin, cos)")
        return df


class RainfallPreprocessor:
    """降雨量预处理器：零膨胀 → 二分类标签"""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def transform(self, df: pd.DataFrame, rain_col: str = 'rainfall',
                  label_col: str = 'rain_flag') -> pd.DataFrame:
        """
        生成降雨二分类标签

        Args:
            df:        DataFrame
            rain_col:  降雨量列名
            label_col: 输出的二分类标签列名

        Returns:
            DataFrame: 添加了 rain_flag 列 (0=无雨, 1=有雨)
        """
        if rain_col not in df.columns:
            logger.warning(f"降雨量列 '{rain_col}' 不存在，跳过转换")
            return df

        df = df.copy()
        df[label_col] = (df[rain_col] >= self.threshold).astype(int)

        zero_ratio = (df[label_col] == 0).mean()
        logger.info(
            f"降雨二分类完成: 阈值={self.threshold}mm, "
            f"无雨 {zero_ratio:.1%}, 有雨 {1 - zero_ratio:.1%}"
        )
        return df


class LightPreprocessor:
    """光照强度预处理器：夜间标记"""

    def __init__(self, night_threshold: float = 100.0):
        self.night_threshold = night_threshold

    def transform(self, df: pd.DataFrame, light_col: str = 'light_intensity',
                  time_col: str = 'timestamp',
                  label_col: str = 'is_night') -> pd.DataFrame:
        """
        生成夜间标记（光照强度低于阈值 或 时间在20:00-06:00之间）

        Args:
            df:         DataFrame
            light_col:  光照强度列名
            time_col:   时间列名
            label_col:  输出标签列名

        Returns:
            DataFrame: 添加了 is_night 列 (0=白天, 1=夜间)
        """
        df = df.copy()

        if light_col in df.columns:
            df[label_col] = (df[light_col] < self.night_threshold).astype(int)
        elif time_col in df.columns:
            hour = pd.to_datetime(df[time_col]).dt.hour
            df[label_col] = ((hour < 6) | (hour >= 20)).astype(int)
        else:
            logger.warning("无法生成夜间标记：光照强度和时间列均不存在")
            return df

        night_ratio = df[label_col].mean()
        logger.info(f"夜间标记完成: 阈值={self.night_threshold}lux, 夜间占比 {night_ratio:.1%}")
        return df


# ═══════════════════════════════════════════════════════════════
# 4. 特征工程类（15分钟间隔优化）
# ═══════════════════════════════════════════════════════════════

class FeatureEngineer15min:
    """
    针对15分钟间隔优化的特征工程

    生成三类特征：
      1. 滞后特征：短期(lag_1,2,4)、中期(lag_96,192)、长期(lag_672)
      2. 时间特征：hour, minute, day_of_week, month, is_daytime, hour_sin/cos
      3. 滚动统计特征：rolling_mean_4/16, rolling_std_4
    """

    def __init__(self,
                 lag_short: List[int] = None,
                 lag_medium: List[int] = None,
                 lag_long: List[int] = None,
                 rolling_mean_windows: List[int] = None,
                 rolling_std_windows: List[int] = None,
                 rolling_min_windows: List[int] = None,
                 rolling_max_windows: List[int] = None):
        self.lag_short = lag_short or [1, 2, 4]
        self.lag_medium = lag_medium or [96, 192]
        self.lag_long = lag_long or [672]
        self.rolling_mean_windows = rolling_mean_windows or [4, 16]
        self.rolling_std_windows = rolling_std_windows or [4]
        self.rolling_min_windows = rolling_min_windows or []
        self.rolling_max_windows = rolling_max_windows or []

    def add_lag_features(self, df: pd.DataFrame,
                         columns: List[str],
                         time_col: str = 'timestamp') -> pd.DataFrame:
        """
        添加滞后特征

        Args:
            df:       DataFrame（已按时间排序）
            columns:  需要生成滞后特征的列
            time_col: 时间列名

        Returns:
            DataFrame: 添加了 lag_{n}_{col} 列
        """
        df = df.copy()
        all_lags = self.lag_short + self.lag_medium + self.lag_long
        added = 0

        for col in columns:
            if col not in df.columns:
                continue
            for lag in all_lags:
                lag_col = f"lag_{lag}_{col}"
                df[lag_col] = df[col].shift(lag)
                added += 1

        logger.info(f"滞后特征: 添加 {added} 列 (lags={all_lags})")
        return df

    def add_time_features(self, df: pd.DataFrame,
                          time_col: str = 'timestamp',
                          light_col: str = 'light_intensity',
                          night_threshold: float = 100.0) -> pd.DataFrame:
        """
        添加时间特征

        Args:
            df:              DataFrame
            time_col:        时间列名
            light_col:       光照强度列名（用于判断白天/夜间）
            night_threshold: 夜间光照阈值

        Returns:
            DataFrame: 添加了时间特征
        """
        df = df.copy()
        ts = pd.to_datetime(df[time_col])

        df['hour'] = ts.dt.hour
        df['minute'] = ts.dt.minute
        df['day_of_week'] = ts.dt.dayofweek
        df['month'] = ts.dt.month

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).round(6)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).round(6)

        if light_col in df.columns:
            df['is_daytime'] = (df[light_col] >= night_threshold).astype(int)
        else:
            df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] < 20)).astype(int)

        logger.info("时间特征: hour, minute, day_of_week, month, is_daytime, hour_sin/cos")
        return df

    def add_rolling_features(self, df: pd.DataFrame,
                             columns: List[str]) -> pd.DataFrame:
        """
        添加滚动统计特征

        Args:
            df:       DataFrame（已按时间排序）
            columns:  需要生成滚动特征的列

        Returns:
            DataFrame: 添加了 rolling_mean_{w}_{col}, rolling_std_{w}_{col} 等
        """
        df = df.copy()
        added = 0

        for col in columns:
            if col not in df.columns:
                continue
            for w in self.rolling_mean_windows:
                feat_name = f"rolling_mean_{w}_{col}"
                df[feat_name] = df[col].rolling(window=w, min_periods=1).mean()
                added += 1
            for w in self.rolling_std_windows:
                feat_name = f"rolling_std_{w}_{col}"
                df[feat_name] = df[col].rolling(window=w, min_periods=1).std()
                added += 1
            for w in self.rolling_min_windows:
                feat_name = f"rolling_min_{w}_{col}"
                df[feat_name] = df[col].rolling(window=w, min_periods=1).min()
                added += 1
            for w in self.rolling_max_windows:
                feat_name = f"rolling_max_{w}_{col}"
                df[feat_name] = df[col].rolling(window=w, min_periods=1).max()
                added += 1

        logger.info(f"滚动统计特征: 添加 {added} 列")
        return df

    def fit_transform(self, df: pd.DataFrame,
                      feature_columns: List[str],
                      time_col: str = 'timestamp',
                      light_col: str = 'light_intensity') -> pd.DataFrame:
        """
        一站式特征工程

        Args:
            df:              DataFrame
            feature_columns: 需要生成特征的列
            time_col:        时间列名
            light_col:       光照强度列名

        Returns:
            DataFrame: 包含所有特征的完整数据
        """
        logger.info("=" * 50)
        logger.info("开始特征工程...")
        logger.info("=" * 50)

        df = self.add_time_features(df, time_col=time_col, light_col=light_col)
        df = self.add_lag_features(df, columns=feature_columns, time_col=time_col)
        df = self.add_rolling_features(df, columns=feature_columns)

        logger.info(f"特征工程完成: {df.shape[1]} 列, {df.shape[0]} 行")
        return df


# ═══════════════════════════════════════════════════════════════
# 5. 多目标关联特征
# ═══════════════════════════════════════════════════════════════

class CrossFeatureGenerator:
    """生成跨目标关联特征"""

    @staticmethod
    def generate(df: pd.DataFrame,
                 greenhouse_temp: str = 'temperature',
                 outdoor_temp: str = 'temperature_outdoor',
                 greenhouse_hum: str = 'humidity',
                 outdoor_hum: str = 'humidity_outdoor') -> pd.DataFrame:
        """
        生成关联特征：
          - temp_diff: 大棚温度 - 户外温度
          - humidity_diff: 大棚湿度 - 户外湿度
          - temp_humidity_interaction: 大棚温度 × 大棚湿度

        Returns:
            DataFrame: 添加了关联特征
        """
        df = df.copy()
        added = []

        if greenhouse_temp in df.columns and outdoor_temp in df.columns:
            df['temp_diff'] = (df[greenhouse_temp] - df[outdoor_temp]).round(3)
            added.append('temp_diff')

        if greenhouse_hum in df.columns and outdoor_hum in df.columns:
            df['humidity_diff'] = (df[greenhouse_hum] - df[outdoor_hum]).round(3)
            added.append('humidity_diff')

        if greenhouse_temp in df.columns and greenhouse_hum in df.columns:
            df['temp_humidity_interaction'] = (
                df[greenhouse_temp] * df[greenhouse_hum]
            ).round(3)
            added.append('temp_humidity_interaction')

        if added:
            logger.info(f"关联特征: {added}")
        else:
            logger.warning("未能生成关联特征：所需列不存在")

        return df


# ═══════════════════════════════════════════════════════════════
# 6. 数据导出器
# ═══════════════════════════════════════════════════════════════

class DataExporter:
    """
    数据导出器：按目标独立导出 X/y，时间序列划分数据集

    AutoResearch 兼容格式：
      每个目标一个目录，包含 X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv
    """

    def __init__(self,
                 output_dir: str = "preprocessed_data",
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 predict_horizon: int = 1):
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.predict_horizon = predict_horizon

    def _time_series_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按时间顺序划分"""
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def _build_xy(self, df: pd.DataFrame, target: str,
                  feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """构建 X, y"""
        if target not in df.columns:
            raise ValueError(f"目标列 '{target}' 不存在")

        y_col = f"y_future_{self.predict_horizon}"
        df = df.copy()
        df[y_col] = df[target].shift(-self.predict_horizon)
        df = df.dropna(subset=[y_col])

        if feature_columns is None:
            exclude = [target, y_col, 'timestamp', 'location']
            feature_columns = [c for c in df.columns if c not in exclude]

        X = df[feature_columns].copy()
        y = df[y_col].copy()

        return X, y

    def export_target(self, df: pd.DataFrame, target: str,
                      feature_columns: Optional[List[str]] = None):
        """
        导出单个目标的数据集

        Args:
            df:              完整特征DataFrame
            target:           目标列名
            feature_columns:  特征列列表（None则自动选择）
        """
        required = ['timestamp'] if 'timestamp' in df.columns else []
        train_df, val_df, test_df = self._time_series_split(df)

        X_train, y_train = self._build_xy(train_df, target, feature_columns)
        X_val, y_val = self._build_xy(val_df, target, feature_columns)
        X_test, y_test = self._build_xy(test_df, target, feature_columns)

        target_dir = os.path.join(self.output_dir, target)
        os.makedirs(target_dir, exist_ok=True)

        X_train.to_csv(os.path.join(target_dir, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(target_dir, 'y_train.csv'), index=False, header=['y'])
        X_val.to_csv(os.path.join(target_dir, 'X_val.csv'), index=False)
        y_val.to_csv(os.path.join(target_dir, 'y_val.csv'), index=False, header=['y'])
        X_test.to_csv(os.path.join(target_dir, 'X_test.csv'), index=False)
        y_test.to_csv(os.path.join(target_dir, 'y_test.csv'), index=False, header=['y'])

        logger.info(
            f"导出 {target}: "
            f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
        )

    def export_all(self, df: pd.DataFrame, targets: List[str],
                   feature_columns: Optional[List[str]] = None):
        """批量导出所有目标"""
        for target in targets:
            if target in df.columns:
                self.export_target(df, target, feature_columns)
            else:
                logger.warning(f"跳过目标 '{target}'：列不存在")


# ═══════════════════════════════════════════════════════════════
# 7. 预处理流水线
# ═══════════════════════════════════════════════════════════════

class PreprocessingPipeline:
    """
    预处理流水线：一站式完成从原始数据到训练数据集的全流程

    使用示例:
        pipeline = PreprocessingPipeline(output_dir="preprocessed_data")

        df_greenhouse = greenhouse_loader.df_standardized
        df_outdoor = outdoor_loader.df_standardized

        df_features = pipeline.run(df_greenhouse, df_outdoor)
        pipeline.export(df_features)
    """

    def __init__(self, config: Optional[Dict] = None, output_dir: str = "preprocessed_data"):
        """
        Args:
            config:     配置字典（不提供则使用 preprocessing_config）
            output_dir: 导出目录
        """
        if config is None:
            from preprocessing_config import PREPROCESS_CONFIG
            config = PREPROCESS_CONFIG

        self.config = config
        self.output_dir = output_dir

        tc = config.get('time', {})
        self.time_aligner = TimeAligner(freq_minutes=tc.get('freq_minutes', 15))

        cc = config.get('cleaning', {})
        self.cleaner = DataCleaner(
            iqr_factor=cc.get('iqr_factor', 1.5),
            sigma_factor=cc.get('sigma_factor', 3.0)
        )

        fc = config.get('features', {})
        lag_cfg = fc.get('lag', {})
        rolling_cfg = fc.get('rolling', {})
        self.feature_engineer = FeatureEngineer15min(
            lag_short=lag_cfg.get('short', [1, 2, 4]),
            lag_medium=lag_cfg.get('medium', [96, 192]),
            lag_long=lag_cfg.get('long', [672]),
            rolling_mean_windows=rolling_cfg.get('mean', [4, 16]),
            rolling_std_windows=rolling_cfg.get('std', [4]),
            rolling_min_windows=rolling_cfg.get('min', []),
            rolling_max_windows=rolling_cfg.get('max', [])
        )

        sp = config.get('special_preprocessors', {})
        self.wind_prep = WindDirectionPreprocessor()
        self.rain_prep = RainfallPreprocessor(
            threshold=sp.get('rainfall', {}).get('binary_threshold', 0.1)
        )
        self.light_prep = LightPreprocessor(
            night_threshold=sp.get('light_intensity', {}).get('night_threshold', 100.0)
        )

        self.cross_generator = CrossFeatureGenerator()

        ec = config.get('export', {})
        split_cfg = ec.get('split', {})
        self.exporter = DataExporter(
            output_dir=output_dir,
            train_ratio=split_cfg.get('train_ratio', 0.8),
            val_ratio=split_cfg.get('val_ratio', 0.1),
            test_ratio=split_cfg.get('test_ratio', 0.1),
            predict_horizon=ec.get('predict_horizon', 1)
        )

        self.df_features = None

    def merge_sources(self, df_greenhouse: pd.DataFrame,
                      df_outdoor: pd.DataFrame) -> pd.DataFrame:
        """
        合并大棚和户外数据（按时间对齐合并为宽表）

        Args:
            df_greenhouse: 大棚数据 (timestamp, temperature, humidity, location)
            df_outdoor:    户外数据 (timestamp, temperature_outdoor, ...)

        Returns:
            DataFrame: 合并后的宽表
        """
        df_g = df_greenhouse.copy()
        df_o = df_outdoor.copy()

        g_cols = {c: c for c in ['temperature', 'humidity'] if c in df_g.columns}
        o_cols = {c: c for c in ['temperature_outdoor', 'humidity_outdoor',
                                   'light_intensity', 'wind_direction',
                                   'wind_speed', 'rainfall'] if c in df_o.columns}

        df_g_sel = df_g[['timestamp'] + list(g_cols.keys())].copy()
        df_o_sel = df_o[['timestamp'] + list(o_cols.keys())].copy()

        df_merged = pd.merge(df_g_sel, df_o_sel, on='timestamp', how='outer')
        df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)

        logger.info(
            f"数据合并完成: 大棚 {len(df_g_sel)} + 户外 {len(df_o_sel)} → {len(df_merged)} 条"
        )
        return df_merged

    def run(self, df_greenhouse: pd.DataFrame,
            df_outdoor: Optional[pd.DataFrame] = None,
            df_merged: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        执行完整预处理流水线

        Args:
            df_greenhouse: 大棚标准化数据
            df_outdoor:    户外标准化数据
            df_merged:     已合并的数据（如果已提前合并）

        Returns:
            DataFrame: 处理完成的特征数据
        """
        if df_merged is not None:
            df = df_merged.copy()
        elif df_outdoor is not None:
            df = self.merge_sources(df_greenhouse, df_outdoor)
        else:
            df = df_greenhouse.copy()

        print("\n" + "=" * 60)
        print("  数据预处理流水线")
        print("=" * 60)

        tc = self.config.get('time', {})
        df = self.time_aligner.align(
            df,
            fill_method=tc.get('fill_method', 'linear'),
            duplicate_method=tc.get('duplicate_method', 'mean')
        )

        cc = self.config.get('cleaning', {})
        df, _ = self.cleaner.clean(
            df,
            anomaly_method=cc.get('anomaly_method', 'iqr'),
            missing_method=cc.get('missing_method', 'linear'),
            smoothing=cc.get('smoothing', False),
            smoothing_window=cc.get('smoothing_window', 3)
        )

        sp = self.config.get('special_preprocessors', {})

        if sp.get('wind_direction', {}).get('enabled', True):
            df = self.wind_prep.transform(df)

        if sp.get('rainfall', {}).get('enabled', True):
            df = self.rain_prep.transform(df)

        if sp.get('light_intensity', {}).get('enabled', True):
            df = self.light_prep.transform(df)

        cf = self.config.get('cross_features', {})
        if any(cf.values()):
            df = self.cross_generator.generate(df)

        feature_columns = [c for c in df.select_dtypes(include=[np.number]).columns
                           if c.startswith(('temperature', 'humidity', 'light_intensity',
                                            'wind_direction', 'wind_speed', 'rainfall'))]
        df = self.feature_engineer.fit_transform(df, feature_columns=feature_columns)

        self.df_features = df

        print("=" * 60)
        print(f"  预处理完成: {df.shape[1]} 列 × {df.shape[0]} 行")
        print("=" * 60)

        return df

    def export(self, targets: Optional[List[str]] = None,
               feature_columns: Optional[List[str]] = None):
        """
        导出训练数据集

        Args:
            targets:        目标列表（None则导出所有配置的目标）
            feature_columns: 特征列列表
        """
        if self.df_features is None:
            raise ValueError("请先调用 run() 方法")

        if targets is None:
            targets = self.config.get('export', {}).get('targets', [])

        self.exporter.export_all(self.df_features, targets, feature_columns)


def main():
    """
    使用示例：演示完整预处理流水线
    """
    from greenhouse_data_loader import GreenhouseDataLoader
    from outdoor_weather_loader import OutdoorWeatherLoader

    current_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  数据预处理模块 - 功能演示")
    print("=" * 60)

    print("\n[1] 加载数据")
    print("-" * 40)
    gh_loader = GreenhouseDataLoader(
        os.path.join(current_dir, '天气数据', '温湿度数据', '温湿度数据.csv')
    )
    gh_loader.load()
    gh_loader.check_quality()
    df_greenhouse = gh_loader.standardize()

    ow_loader = OutdoorWeatherLoader(
        os.path.join(current_dir, '天气数据', '宣威市尚营种气象', '宣威市尚营种气象.csv')
    )
    ow_loader.load()
    ow_loader.check_quality()
    df_outdoor = ow_loader.standardize()

    print(f"大棚数据: {df_greenhouse.shape}")
    print(f"户外数据: {df_outdoor.shape}")

    print("\n[2] 合并数据源")
    print("-" * 40)
    pipeline = PreprocessingPipeline(output_dir=os.path.join(current_dir, 'preprocessed_data'))
    df_merged = pipeline.merge_sources(df_greenhouse, df_outdoor)
    print(f"合并后: {df_merged.shape}")
    print(df_merged.head().to_string())

    print("\n[3] 时间对齐")
    print("-" * 40)
    df_aligned = pipeline.time_aligner.align(df_merged)
    print(f"对齐后: {df_aligned.shape}")

    print("\n[4] 数据清洗")
    print("-" * 40)
    df_clean, clean_report = pipeline.cleaner.clean(df_aligned)
    print(f"清洗后: {df_clean.shape}")

    print("\n[5] 特殊预处理器")
    print("-" * 40)

    df_feat = df_clean.copy()

    df_feat = pipeline.wind_prep.transform(df_feat)
    df_feat = pipeline.rain_prep.transform(df_feat)
    df_feat = pipeline.light_prep.transform(df_feat)

    df_feat = pipeline.cross_generator.generate(df_feat)

    print(f"特殊处理后: {df_feat.shape}")
    new_cols = [c for c in df_feat.columns if c not in df_clean.columns]
    print(f"新增列: {new_cols}")

    print("\n[6] 特征工程")
    print("-" * 40)
    feature_cols = ['temperature', 'humidity', 'temperature_outdoor',
                    'humidity_outdoor', 'light_intensity', 'wind_speed']
    df_features = pipeline.feature_engineer.fit_transform(df_feat, feature_columns=feature_cols)
    print(f"特征工程后: {df_features.shape}")

    print("\n[7] 数据导出")
    print("-" * 40)
    targets = ['temperature', 'humidity', 'temperature_outdoor']
    pipeline.df_features = df_features
    pipeline.export(targets=targets)

    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()