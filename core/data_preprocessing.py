#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理管线 —— 大棚与室外数据独立处理

策略变更:
  - 不再将大棚数据和室外数据合并插值
  - 大棚数据（~610条，~7天）在自己的时间网格上独立处理
  - 室外数据（~131K条，~5年）在自己的时间网格上独立处理
  - 大棚数据因记录少，不使用长滞后特征（lag_672 不可用）
"""

import os
import sys
import logging
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, '..'))

from configs.preprocessing_config import PREPROCESS_CONFIG, FEATURE_GROUPS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

TARGET_COLUMN_MAP = {
    'temperature': 'temperature',
    'humidity': 'humidity',
    'temperature_outdoor': 'temperature_outdoor',
    'humidity_outdoor': 'humidity_outdoor',
    'light_intensity': 'light_intensity',
    'wind_direction': 'wind_direction',
    'wind_speed': 'wind_speed',
    'rainfall': 'rainfall',
}

GREENHOUSE_COLUMNS = ['temperature', 'humidity']
OUTDOOR_COLUMNS = [
    'temperature_outdoor', 'humidity_outdoor',
    'light_intensity', 'wind_direction', 'wind_speed', 'rainfall'
]


class TimeAligner:
    """时间对齐器 —— 将单个数据集对齐到统一 15min 网格"""

    def __init__(self, freq_minutes=15, fill_method='linear', duplicate_method='mean'):
        self.freq = freq_minutes
        self.fill_method = fill_method
        self.duplicate_method = duplicate_method

    def align(self, df: pd.DataFrame, timestamp_col='timestamp') -> pd.DataFrame:
        """
        单数据集对齐: 创建统一15min网格, 去重, 填充缺失

        与旧版的区别: 不会与外数据集合并, 时间范围仅根据自身数据决定
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.drop_duplicates(subset=[timestamp_col])
        df = df.sort_values(timestamp_col)

        # 自身数据的起止时间
        t_min = df[timestamp_col].min()
        t_max = df[timestamp_col].max()

        # 对齐到 freq 分钟整点
        t_min = t_min.replace(second=0, microsecond=0)
        t_max = t_max.replace(second=0, microsecond=0)
        t_min = t_min - timedelta(minutes=t_min.minute % self.freq)
        t_max = t_max - timedelta(minutes=t_max.minute % self.freq)

        grid = pd.date_range(t_min, t_max, freq=timedelta(minutes=self.freq))
        grid_df = pd.DataFrame({timestamp_col: grid})

        # 处理重复时间点
        if df.duplicated(subset=[timestamp_col]).any():
            agg_dict = {}
            for col in df.columns:
                if col == timestamp_col:
                    continue
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'first'
            df = df.groupby(timestamp_col, as_index=False).agg(agg_dict)

        # 合并到网格
        result = grid_df.merge(df, on=timestamp_col, how='left')

        # 填充缺失
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        if self.fill_method == 'linear':
            for col in numeric_cols:
                if col == timestamp_col:
                    continue
                result[col] = result[col].interpolate(method='linear', limit_direction='both')
        elif self.fill_method == 'ffill':
            for col in numeric_cols:
                result[col] = result[col].fillna(method='ffill').fillna(method='bfill')
        elif self.fill_method == 'zero':
            for col in numeric_cols:
                result[col] = result[col].fillna(0)

        result = result.sort_values(timestamp_col).reset_index(drop=True)
        return result


class DataCleaner:
    """数据清洗器 —— IQR 或 3σ 异常值检测"""

    def __init__(self, anomaly_method='iqr', iqr_factor=1.5, sigma_factor=3.0,
                 missing_method='linear', smoothing=False, smoothing_window=3,
                 smoothing_method='rolling_mean'):
        self.anomaly_method = anomaly_method
        self.iqr_factor = iqr_factor
        self.sigma_factor = sigma_factor
        self.missing_method = missing_method
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window
        self.smoothing_method = smoothing_method

    def process(self, df: pd.DataFrame, time_aligner: TimeAligner = None,
                timestamp_col='timestamp') -> pd.DataFrame:
        if time_aligner is not None:
            df = time_aligner.align(df, timestamp_col)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col == timestamp_col:
                continue
            if self.anomaly_method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.iqr_factor * IQR
                upper = Q3 + self.iqr_factor * IQR
                if IQR > 0:
                    df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
            elif self.anomaly_method == 'sigma':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    lower = mean - self.sigma_factor * std
                    upper = mean + self.sigma_factor * std
                    df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
            elif self.anomaly_method == 'iqr_sigma':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                mean = df[col].mean()
                std = df[col].std()
                if IQR > 0 and std > 0:
                    iqr_lower = Q1 - self.iqr_factor * IQR
                    iqr_upper = Q3 + self.iqr_factor * IQR
                    sigma_lower = mean - self.sigma_factor * std
                    sigma_upper = mean + self.sigma_factor * std
                    df.loc[
                        ((df[col] < iqr_lower) | (df[col] > iqr_upper)) &
                        ((df[col] < sigma_lower) | (df[col] > sigma_upper)),
                        col
                    ] = np.nan

        missing_count = df[numeric_cols].isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"检测到 {missing_count} 个异常值/缺失值，使用 {self.missing_method} 填充")
            if self.missing_method == 'linear':
                for col in numeric_cols:
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
            elif self.missing_method == 'ffill':
                for col in numeric_cols:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        if self.smoothing:
            for col in numeric_cols:
                if col == timestamp_col:
                    continue
                window = max(1, int(self.smoothing_window))
                if self.smoothing_method == 'rolling_mean':
                    df[col] = df[col].rolling(window=window, center=True, min_periods=1).mean()
                elif self.smoothing_method == 'rolling_median':
                    df[col] = df[col].rolling(window=window, center=True, min_periods=1).median()

        return df


class FeatureEngineer15min:
    """15分钟间隔特征工程"""

    def __init__(self, lag_config: dict, time_features: list,
                 rolling_config: dict, cross_features: dict,
                 special_preprocessors: dict = None,
                 max_lag_overflow: bool = False):
        self.lag_config = lag_config
        self.time_features = time_features
        self.rolling_config = rolling_config
        self.cross_features = cross_features
        self.special_preprocessors = special_preprocessors or {}
        self.max_lag_overflow = max_lag_overflow

    def _safe_lag_columns(self, columns: List[str], df_len: int) -> List[int]:
        """过滤掉超过数据长度的滞后值"""
        result = []
        for lag in columns:
            if lag < df_len:
                result.append(lag)
            else:
                if self.max_lag_overflow:
                    logger.debug(f"跳过 lag_{lag}: 数据长度 {df_len} 不足")
        return result

    def process(self, df: pd.DataFrame, timestamp_col='timestamp') -> pd.DataFrame:
        result = df.copy()
        exclude_cols = ['hour', 'minute', 'day_of_week', 'month', 'is_daytime',
                        'hour_sin', 'hour_cos', 'wind_sin', 'wind_cos',
                        'rain_flag', 'is_night', 'temp_diff', 'humidity_diff',
                        'temp_humidity_interaction', 'location']
        target_cols = [c for c in result.columns if c != timestamp_col
                       and c not in exclude_cols
                       and '_lag_' not in c
                       and 'rolling_' not in c
                       and np.issubdtype(result[c].dtype, np.number)]

        all_lags = (
            self.lag_config.get('short', []) +
            self.lag_config.get('medium', []) +
            self.lag_config.get('long', [])
        )
        valid_lags = self._safe_lag_columns(all_lags, len(result))

        for col in target_cols:
            for lag in valid_lags:
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)

        for window in self.rolling_config.get('mean', []):
            for col in target_cols:
                result[f'{col}_rolling_mean_{window}'] = (
                    result[col].rolling(window=window, min_periods=1).mean()
                )

        for window in self.rolling_config.get('std', []):
            for col in target_cols:
                result[f'{col}_rolling_std_{window}'] = (
                    result[col].rolling(window=window, min_periods=1).std().fillna(0)
                )

        for window in self.rolling_config.get('min', []):
            for col in target_cols:
                result[f'{col}_rolling_min_{window}'] = (
                    result[col].rolling(window=window, min_periods=1).min()
                )

        for window in self.rolling_config.get('max', []):
            for col in target_cols:
                result[f'{col}_rolling_max_{window}'] = (
                    result[col].rolling(window=window, min_periods=1).max()
                )

        ts = result[timestamp_col]
        if 'hour' in self.time_features or 'hour_sin' in self.time_features:
            result['hour'] = ts.dt.hour
        if 'minute' in self.time_features:
            result['minute'] = ts.dt.minute
        if 'day_of_week' in self.time_features:
            result['day_of_week'] = ts.dt.dayofweek
        if 'month' in self.time_features:
            result['month'] = ts.dt.month
        if 'is_daytime' in self.time_features:
            hour_val = ts.dt.hour
            result['is_daytime'] = ((hour_val >= 6) & (hour_val < 20)).astype(int)
        if 'hour_sin' in self.time_features:
            result['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
        if 'hour_cos' in self.time_features:
            result['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)

        # 特殊预处理
        if (self.special_preprocessors.get('wind_direction', {}).get('enabled')
                and 'wind_direction' in result.columns):
            wd_rad = np.radians(result['wind_direction'].fillna(0))
            result['wind_sin'] = np.sin(wd_rad)
            result['wind_cos'] = np.cos(wd_rad)

        if (self.special_preprocessors.get('rainfall', {}).get('enabled')
                and 'rainfall' in result.columns):
            threshold = self.special_preprocessors['rainfall'].get('binary_threshold', 0.1)
            label = self.special_preprocessors['rainfall'].get('binary_label', 'rain_flag')
            result[label] = (result['rainfall'] > threshold).astype(int)

        if (self.special_preprocessors.get('light_intensity', {}).get('enabled')
                and 'light_intensity' in result.columns):
            threshold = self.special_preprocessors['light_intensity'].get('night_threshold', 100.0)
            label = self.special_preprocessors['light_intensity'].get('night_label', 'is_night')
            hour_val = ts.dt.hour
            is_night_time = (hour_val < 6) | (hour_val >= 20)
            result[label] = ((result['light_intensity'] < threshold) & is_night_time).astype(int)

        # 交叉特征
        if self.cross_features.get('temp_diff') and 'temperature' in result.columns and 'temperature_outdoor' in result.columns:
            result['temp_diff'] = result['temperature'] - result['temperature_outdoor']
        elif self.cross_features.get('temp_diff'):
            result['temp_diff'] = 0

        if self.cross_features.get('humidity_diff') and 'humidity' in result.columns and 'humidity_outdoor' in result.columns:
            result['humidity_diff'] = result['humidity'] - result['humidity_outdoor']
        elif self.cross_features.get('humidity_diff'):
            result['humidity_diff'] = 0

        if self.cross_features.get('temp_humidity_interaction'):
            temp_col = None
            hum_col = None
            if 'temperature' in result.columns:
                temp_col = 'temperature'
                hum_col = 'humidity' if 'humidity' in result.columns else None
            if temp_col is None and 'temperature_outdoor' in result.columns:
                temp_col = 'temperature_outdoor'
                hum_col = 'humidity_outdoor' if 'humidity_outdoor' in result.columns else None
            if temp_col and hum_col:
                result['temp_humidity_interaction'] = result[temp_col] * result[hum_col]
            else:
                result['temp_humidity_interaction'] = 0

        # 删除 NaN 行（由滞后特征产生）
        drop_len = max(valid_lags) if valid_lags else 0
        if drop_len > 0:
            result = result.iloc[drop_len:].reset_index(drop=True)

        return result


class DataExporter:
    """数据导出器 —— 按目标导出 train/val/test 和特征列表"""

    def __init__(self, output_dir='preprocessed_data'):
        self.output_dir = output_dir

    def export(self, df: pd.DataFrame, targets: List[str],
               target_column_map: dict, all_features: List[str],
               predict_horizon=1, split_config=None, output_format='csv'):
        if split_config is None:
            split_config = {'train_ratio': 0.8, 'val_ratio': 0.1, 'test_ratio': 0.1, 'method': 'time_series'}

        os.makedirs(self.output_dir, exist_ok=True)
        n = len(df)
        train_end = int(n * split_config['train_ratio'])
        val_end = int(n * (split_config['train_ratio'] + split_config['val_ratio']))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        exported = []
        for target_name in targets:
            col_name = target_column_map.get(target_name)
            if col_name is None or col_name not in df.columns:
                logger.warning(f"跳过 {target_name}: 列 '{col_name}' 不存在于数据中")
                continue

            feature_cols = [f for f in all_features if f in df.columns and f != col_name]

            if len(train_df) <= predict_horizon or len(feature_cols) == 0:
                logger.warning(f"跳过 {target_name}: 数据不足或特征为空")
                continue

            target_dir = os.path.join(self.output_dir, target_name)
            os.makedirs(target_dir, exist_ok=True)

            self._save_split(train_df, val_df, test_df, feature_cols, col_name,
                             predict_horizon, target_dir, output_format)
            exported.append(target_name)
            logger.info(f"已导出 {target_name}: train={train_end}, val={val_end - train_end}, test={n - val_end}")

        return exported

    def _save_split(self, train_df, val_df, test_df, feature_cols, target_col,
                    predict_horizon, target_dir, output_format):
        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col].shift(-predict_horizon).iloc[:-predict_horizon]
        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_col].shift(-predict_horizon).iloc[:-predict_horizon]
        X_test = test_df[feature_cols].copy()
        y_test = test_df[target_col].shift(-predict_horizon).iloc[:-predict_horizon]

        X_train = X_train.iloc[:len(y_train)]
        X_val = X_val.iloc[:len(y_val)]
        X_test = X_test.iloc[:len(y_test)]

        if output_format == 'csv':
            X_train.to_csv(os.path.join(target_dir, 'X_train.csv'), index=False)
            pd.DataFrame({'y': y_train.values}).to_csv(
                os.path.join(target_dir, 'y_train.csv'), index=False)
            X_val.to_csv(os.path.join(target_dir, 'X_val.csv'), index=False)
            pd.DataFrame({'y': y_val.values}).to_csv(
                os.path.join(target_dir, 'y_val.csv'), index=False)
            X_test.to_csv(os.path.join(target_dir, 'X_test.csv'), index=False)
            pd.DataFrame({'y': y_test.values}).to_csv(
                os.path.join(target_dir, 'y_test.csv'), index=False)


class PreprocessingPipeline:
    """
    预处理管线 —— 大棚和室外独立处理

    用法:
        gh_path = '../天气数据/温湿度数据/温湿度数据.csv'
        ow_path = '../天气数据/宣威市尚营种气象/宣威市尚营种气象.csv'

        pipeline = PreprocessingPipeline()
        pipeline.run(gh_path, ow_path)
        pipeline.export()   # 导出全部8个目标
    """

    def __init__(self, config=None, output_dir='preprocessed_data'):
        self.cfg = config or PREPROCESS_CONFIG
        self.output_dir = output_dir

        time_cfg = self.cfg['time']
        clean_cfg = self.cfg['cleaning']
        feat_cfg = self.cfg['features']
        export_cfg = self.cfg['export']

        self.time_aligner = TimeAligner(
            freq_minutes=time_cfg.get('freq_minutes', 15),
            fill_method=time_cfg.get('fill_method', 'linear'),
            duplicate_method=time_cfg.get('duplicate_method', 'mean'),
        )

        self.cleaner = DataCleaner(
            anomaly_method=clean_cfg.get('anomaly_method', 'iqr'),
            iqr_factor=clean_cfg.get('iqr_factor', 1.5),
            sigma_factor=clean_cfg.get('sigma_factor', 3.0),
            missing_method=clean_cfg.get('missing_method', 'linear'),
            smoothing=clean_cfg.get('smoothing', False),
            smoothing_window=clean_cfg.get('smoothing_window', 3),
            smoothing_method=clean_cfg.get('smoothing_method', 'rolling_mean'),
        )

        self.split_config = export_cfg.get('split', {})
        self.predict_horizon = export_cfg.get('predict_horizon', 1)

        self.greenhouse_df = None
        self.outdoor_df = None
        self.feature_columns = []

    def run(self, greenhouse_path: str = None, outdoor_path: str = None,
            greenhouse_df: pd.DataFrame = None, outdoor_df: pd.DataFrame = None):
        """
        运行预处理管线: 大棚和室外各自独立处理

        支持传入 CSV 路径或已加载的 DataFrame
        """
        special_cfg = self.cfg.get('special_preprocessors', {})
        feat_cfg = self.cfg['features']
        cross_cfg = self.cfg.get('cross_features', {})

        # ──── 处理室外数据 ────
        if outdoor_df is None and outdoor_path is not None:
            from outdoor_weather_loader import OutdoorWeatherLoader
            ow_loader = OutdoorWeatherLoader(outdoor_path)
            ow_loader.load()
            ow_loader.check_quality()
            outdoor_df = ow_loader.standardize()

        if outdoor_df is not None:
            logger.info(f"处理室外数据: {len(outdoor_df)} 条原始记录")
            ow_aligned = self.time_aligner.align(outdoor_df)
            ow_clean = self.cleaner.process(ow_aligned, time_aligner=None)

            outdoor_engineer = FeatureEngineer15min(
                lag_config=feat_cfg['lag'],
                time_features=feat_cfg['time_features'],
                rolling_config=feat_cfg['rolling'],
                cross_features=cross_cfg,
                special_preprocessors=special_cfg,
                max_lag_overflow=False,
            )
            self.outdoor_df = outdoor_engineer.process(ow_clean)
            logger.info(f"室外数据处理完成: {len(self.outdoor_df)} 条, {len(self.outdoor_df.columns)} 列")

        # ──── 处理大棚数据 ────
        if greenhouse_df is None and greenhouse_path is not None:
            from greenhouse_data_loader import GreenhouseDataLoader
            gh_loader = GreenhouseDataLoader(greenhouse_path)
            gh_loader.load()
            gh_loader.check_quality()
            greenhouse_df = gh_loader.standardize()

        if greenhouse_df is not None:
            logger.info(f"处理大棚数据: {len(greenhouse_df)} 条原始记录")
            gh_aligned = self.time_aligner.align(greenhouse_df)
            gh_clean = self.cleaner.process(gh_aligned, time_aligner=None)

            # 大棚数据量小，调整滞后配置
            gh_lag_config = {
                'short': feat_cfg['lag'].get('short', [1, 2, 4]),
                'medium': [],   # 数据不足7天，不使用中长滞后
                'long': [],
            }
            # 如果数据足够，启用 1 天滞后 (lag_96)
            if len(gh_clean) > 96:
                gh_lag_config['medium'] = [96]

            greenhouse_engineer = FeatureEngineer15min(
                lag_config=gh_lag_config,
                time_features=feat_cfg['time_features'],
                rolling_config={'mean': [4], 'std': [4]},  # 减小滚动窗口
                cross_features=cross_cfg,
                special_preprocessors={},  # 大棚数据无风向/降雨
                max_lag_overflow=True,
            )
            self.greenhouse_df = greenhouse_engineer.process(gh_clean)
            logger.info(f"大棚数据处理完成: {len(self.greenhouse_df)} 条, {len(self.greenhouse_df.columns)} 列")

        # ──── 收集所有特征列 ────
        all_features = set()
        for df in [self.outdoor_df, self.greenhouse_df]:
            if df is not None:
                for col in df.columns:
                    if col != 'timestamp' and col != 'location':
                        all_features.add(col)
        self.feature_columns = sorted(all_features)
        logger.info(f"总特征数: {len(self.feature_columns)}")

    def export(self, targets=None):
        """
        导出所有目标数据

        目标分配:
          - 大棚数据 → greenhouse_temperature, greenhouse_humidity
          - 室外数据 → outdoor_temperature, outdoor_humidity,
                        light_intensity, wind_direction, wind_speed, rainfall
        """
        if targets is None:
            targets = self.cfg['export'].get('targets', list(TARGET_COLUMN_MAP.keys()))

        exporter = DataExporter(self.output_dir)

        greenhouse_targets = [t for t in targets if TARGET_COLUMN_MAP.get(t) in GREENHOUSE_COLUMNS]
        outdoor_targets = [t for t in targets if TARGET_COLUMN_MAP.get(t) in OUTDOOR_COLUMNS]

        all_exported = []

        if self.greenhouse_df is not None and greenhouse_targets:
            logger.info(f"导出大棚目标: {greenhouse_targets}")
            gh_features = [f for f in self.greenhouse_df.columns
                           if f not in ('timestamp', 'location')]
            exported = exporter.export(
                self.greenhouse_df, greenhouse_targets,
                TARGET_COLUMN_MAP, gh_features,
                predict_horizon=self.predict_horizon,
                split_config=self.split_config,
            )
            all_exported.extend(exported)

        if self.outdoor_df is not None and outdoor_targets:
            logger.info(f"导出室外目标: {outdoor_targets}")
            ow_features = [f for f in self.outdoor_df.columns
                           if f not in ('timestamp', 'location')]
            exported = exporter.export(
                self.outdoor_df, outdoor_targets,
                TARGET_COLUMN_MAP, ow_features,
                predict_horizon=self.predict_horizon,
                split_config=self.split_config,
            )
            all_exported.extend(exported)

        logger.info(f"全部导出完成: {len(all_exported)} 个目标: {all_exported}")
        return all_exported


def main():
    """独立运行预处理管线"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    gh_path = os.path.join(current_dir, '天气数据', '温湿度数据', '温湿度数据.csv')
    ow_path = os.path.join(current_dir, '天气数据', '宣威市尚营种气象', '宣威市尚营种气象.csv')

    pipeline = PreprocessingPipeline(
        output_dir=os.path.join(current_dir, 'preprocessed_data')
    )

    pipeline.run(greenhouse_path=gh_path, outdoor_path=ow_path)
    pipeline.export()


if __name__ == "__main__":
    main()