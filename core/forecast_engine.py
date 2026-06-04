#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
递归多步预测引擎 —— 支持预测未来 24 小时（96 步，每步 15 分钟）

原理:
  XGBoost 模型是单步预测器（predict_horizon=1），要实现多步预测需要递归：
  1. 用当前特征预测 t+1 时刻
  2. 将预测值当作"已知数据"追加到时间序列
  3. 在扩展后的数据上重新生成特征
  4. 用新特征预测 t+2 时刻
  5. 重复直到达到目标步数

为每步重新运行完整 FeatureEngineer15min 管线，确保特征一致性。
"""

import os
import sys
import logging
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

GREENHOUSE_TARGETS = ['greenhouse_temperature', 'greenhouse_humidity']
OUTDOOR_TARGETS = [
    'outdoor_temperature', 'outdoor_humidity',
    'light_intensity', 'wind_direction', 'wind_speed', 'rainfall'
]

TARGET_TO_INTERNAL = {
    'greenhouse_temperature': 'temperature',
    'greenhouse_humidity': 'humidity',
    'outdoor_temperature': 'temperature_outdoor',
    'outdoor_humidity': 'humidity_outdoor',
    'light_intensity': 'light_intensity',
    'wind_direction': 'wind_direction',
    'wind_speed': 'wind_speed',
    'rainfall': 'rainfall',
}

INTERNAL_TO_TARGET = {v: k for k, v in TARGET_TO_INTERNAL.items()}

TARGET_LABELS = {
    'greenhouse_temperature': '大棚温度', 'greenhouse_humidity': '大棚湿度',
    'outdoor_temperature': '室外温度', 'outdoor_humidity': '室外湿度',
    'light_intensity': '光照强度', 'wind_direction': '风向',
    'wind_speed': '风速', 'rainfall': '降雨量',
}

TARGET_UNITS = {
    'greenhouse_temperature': '°C', 'greenhouse_humidity': '%',
    'outdoor_temperature': '°C', 'outdoor_humidity': '%',
    'light_intensity': 'lux', 'wind_direction': '°',
    'wind_speed': 'm/s', 'rainfall': 'mm',
}


def _make_outdoor_engineer():
    """创建与训练时完全一致的室外特征工程器"""
    from configs.preprocessing_config import PREPROCESS_CONFIG
    from core.data_preprocessing import FeatureEngineer15min
    cfg = PREPROCESS_CONFIG
    return FeatureEngineer15min(
        lag_config=cfg['features']['lag'],
        time_features=cfg['features']['time_features'],
        rolling_config=cfg['features']['rolling'],
        cross_features=cfg.get('cross_features', {}),
        special_preprocessors=cfg.get('special_preprocessors', {}),
        max_lag_overflow=False,
    )


def _make_greenhouse_engineer(data_len: int):
    """创建与训练时完全一致的大棚特征工程器"""
    from configs.preprocessing_config import PREPROCESS_CONFIG
    from core.data_preprocessing import FeatureEngineer15min
    cfg = PREPROCESS_CONFIG
    gh_lag_config = {
        'short': cfg['features']['lag'].get('short', [1, 2, 4]),
        'medium': [96] if data_len > 96 else [],
        'long': [],
    }
    return FeatureEngineer15min(
        lag_config=gh_lag_config,
        time_features=cfg['features']['time_features'],
        rolling_config={'mean': [4], 'std': [4]},
        cross_features=cfg.get('cross_features', {}),
        special_preprocessors={},
        max_lag_overflow=True,
    )


class RecursiveForecaster:
    """
    递归多步预测器

    用法:
        from predict_now import ModelLoader
        loader = ModelLoader(); loader.load_all()
        forecaster = RecursiveForecaster(loader)

        # 传入已有的 featured DataFrame（来自 PreprocessingPipeline）
        forecast = forecaster.forecast_outdoor(featured_outdoor_df, steps=96)
        # forecast['outdoor_temperature'] = [(ts1, val1), (ts2, val2), ...]
    """

    def __init__(self, loader):
        self.loader = loader

    def _verify_model_loaded(self, target_name):
        if target_name not in self.loader.models:
            raise KeyError(f"模型未加载: {target_name}")

    def _predict_one(self, target_name, features_row: pd.Series):
        self._verify_model_loaded(target_name)
        model_data = self.loader.models[target_name]
        meta = model_data.get('meta', {})
        target_type = meta.get('target_type', 'regression')
        feature_cols = meta.get('feature_columns', [])

        X = pd.DataFrame([features_row[feature_cols].values], columns=feature_cols)

        if target_type == 'regression':
            return float(model_data['model'].predict(X)[0])
        elif target_type == 'angular_regression':
            sin_pred = float(model_data['model_sin'].predict(X)[0])
            cos_pred = float(model_data['model_cos'].predict(X)[0])
            return float(np.degrees(np.arctan2(sin_pred, cos_pred)) % 360)
        elif target_type == 'two_stage':
            rain_prob = float(model_data['cls_model'].predict_proba(X)[0, 1])
            if rain_prob >= 0.5 and model_data.get('reg_model') is not None:
                return float(max(0, model_data['reg_model'].predict(X)[0]))
            return 0.0

    def forecast_outdoor(self, featured_df: pd.DataFrame, steps: int = 96, latest_timestamp=None) -> dict:
        """
        递归预测室外未来 N 步

        Args:
            featured_df: 室外 Featured DataFrame（PreprocessingPipeline 输出，含所有特征列）
            steps: 预测步数（默认 96 = 24小时）
            latest_timestamp: 原始数据的最新时间（可选），用于修正预测起点

        Returns:
            {target_name: [(timestamp, value), ...]}
        """
        engine = _make_outdoor_engineer()
        internal_cols = [TARGET_TO_INTERNAL[t] for t in OUTDOOR_TARGETS]

        raw_df = featured_df[['timestamp'] + internal_cols].copy()
        
        # 如果提供了原始数据的最新时间，用它来修正预测起点
        if latest_timestamp is not None:
            last_ts = pd.Timestamp(latest_timestamp)
            # 计算管线数据与实际数据的偏移量
            featured_last = raw_df['timestamp'].iloc[-1]
            offset = last_ts - featured_last
            logger.info(f"室外预测: 管线终点={featured_last}, 原始数据终点={last_ts}, 时间偏移={offset}")
        else:
            last_ts = raw_df['timestamp'].iloc[-1]
            offset = timedelta(0)
        
        logger.info(f"室外递归预测起点: {last_ts}, 步数: {steps}")

        results = {t: [] for t in OUTDOOR_TARGETS}

        for step in range(steps):
            next_ts = last_ts + timedelta(minutes=15 * (step + 1))

            featured = engine.process(raw_df.copy())
            last_row = featured.iloc[-1]

            predictions = {}
            for target_name in OUTDOOR_TARGETS:
                try:
                    val = self._predict_one(target_name, last_row)
                except Exception as e:
                    logger.warning(f"预测 {target_name} 第{step+1}步失败: {e}")
                    val = raw_df[TARGET_TO_INTERNAL[target_name]].iloc[-1]
                predictions[TARGET_TO_INTERNAL[target_name]] = val
                results[target_name].append({
                    'step': step + 1,
                    'timestamp': str(next_ts),
                    'value': round(val, 4),
                })

            new_row = {'timestamp': next_ts}
            new_row.update(predictions)
            raw_df = pd.concat([raw_df, pd.DataFrame([new_row])], ignore_index=True)

        return results

    def forecast_greenhouse(self, featured_df: pd.DataFrame, steps: int = 96, latest_timestamp=None) -> dict:
        """
        递归预测大棚未来 N 步
        """
        internal_cols = [TARGET_TO_INTERNAL[t] for t in GREENHOUSE_TARGETS]
        raw_df = featured_df[['timestamp'] + internal_cols].copy()
        
        # 如果提供了原始数据的最新时间，用它来修正预测起点
        if latest_timestamp is not None:
            last_ts = pd.Timestamp(latest_timestamp)
        else:
            last_ts = raw_df['timestamp'].iloc[-1]
        
        engine = _make_greenhouse_engineer(len(raw_df) + steps)
        logger.info(f"大棚递归预测起点: {last_ts}, 步数: {steps}")

        results = {t: [] for t in GREENHOUSE_TARGETS}
        col_has = set(raw_df.columns)

        for step in range(steps):
            next_ts = last_ts + timedelta(minutes=15 * (step + 1))

            featured = engine.process(raw_df.copy())
            last_row = featured.iloc[-1]

            new_row = {'timestamp': next_ts}
            for target_name in GREENHOUSE_TARGETS:
                try:
                    val = self._predict_one(target_name, last_row)
                except Exception as e:
                    logger.warning(f"预测 {target_name} 第{step+1}步失败: {e}")
                    val = raw_df[TARGET_TO_INTERNAL[target_name]].iloc[-1]
                new_row[TARGET_TO_INTERNAL[target_name]] = val
                results[target_name].append({
                    'step': step + 1,
                    'timestamp': str(next_ts),
                    'value': round(val, 4),
                })

            raw_df = pd.concat([raw_df, pd.DataFrame([new_row])], ignore_index=True)

        return results

    def forecast_all(self, greenhouse_df: pd.DataFrame, outdoor_df: pd.DataFrame,
                     steps: int = 96, latest_gh_timestamp=None, latest_ow_timestamp=None) -> dict:
        """
        预测全部目标未来 N 步

        Args:
            latest_gh_timestamp: 大棚原始数据的最新时间
            latest_ow_timestamp: 室外原始数据的最新时间

        Returns:
            {
                'outdoor': {target: [{step, timestamp, value}, ...]},
                'greenhouse': {target: [{step, timestamp, value}, ...]},
            }
        """
        logger.info(f"开始递归预测 {steps} 步（{steps * 15 / 60:.1f} 小时）...")

        ow = self.forecast_outdoor(outdoor_df, steps, latest_timestamp=latest_ow_timestamp) if outdoor_df is not None else {}
        gh = self.forecast_greenhouse(greenhouse_df, steps, latest_timestamp=latest_gh_timestamp) if greenhouse_df is not None else {}

        logger.info(f"递归预测完成: 室外 {len(ow)} 目标, 大棚 {len(gh)} 目标, 各 {steps} 步")
        return {'outdoor': ow, 'greenhouse': gh}

    def forecast_to_timeline(self, greenhouse_df: pd.DataFrame, outdoor_df: pd.DataFrame,
                             steps: int = 96, latest_gh_timestamp=None, latest_ow_timestamp=None) -> list:
        """
        预测并返回统一时间线格式（前端友好）
        注意：室外和大棚数据使用各自最新时间作为预测起点

        Args:
            greenhouse_df: 大棚特征工程处理后的 DataFrame
            outdoor_df: 室外特征工程处理后的 DataFrame
            steps: 预测步数
            latest_gh_timestamp: 大棚原始数据的最新时间（可选）
            latest_ow_timestamp: 室外原始数据的最新时间（可选）

        Returns:
            {
                'outdoor': {
                    'latest_timestamp': '2026-04-30 19:20:31',
                    'timeline': [{'step': 1, 'timestamp': '...', 'predictions': {...}}, ...]
                },
                'greenhouse': {
                    'latest_timestamp': '2026-05-08 17:30:00',
                    'timeline': [{'step': 1, 'timestamp': '...', 'predictions': {...}}, ...]
                }
            }
        """
        all_results = self.forecast_all(greenhouse_df, outdoor_df, steps,
                                         latest_gh_timestamp=latest_gh_timestamp,
                                         latest_ow_timestamp=latest_ow_timestamp)

        result = {}

        # 处理室外预测
        if all_results.get('outdoor'):
            ow_timeline = []
            first_ow = list(all_results['outdoor'].values())[0]
            for i in range(steps):
                if i < len(first_ow):
                    step_data = {
                        'step': i + 1,
                        'timestamp': first_ow[i]['timestamp'],
                        'predictions': {}
                    }
                    for t, vals in all_results['outdoor'].items():
                        step_data['predictions'][t] = vals[i]['value']
                    ow_timeline.append(step_data)
            result['outdoor'] = {
                'latest_timestamp': str(latest_ow_timestamp) if latest_ow_timestamp else None,
                'timeline': ow_timeline
            }

        # 处理大棚预测
        if all_results.get('greenhouse'):
            gh_timeline = []
            first_gh = list(all_results['greenhouse'].values())[0]
            for i in range(steps):
                if i < len(first_gh):
                    step_data = {
                        'step': i + 1,
                        'timestamp': first_gh[i]['timestamp'],
                        'predictions': {}
                    }
                    for t, vals in all_results['greenhouse'].items():
                        step_data['predictions'][t] = vals[i]['value']
                    gh_timeline.append(step_data)
            result['greenhouse'] = {
                'latest_timestamp': str(latest_gh_timestamp) if latest_gh_timestamp else None,
                'timeline': gh_timeline
            }

        return result


def build_forecast_summary(timeline):
    """从时间线提取摘要统计"""
    if not timeline:
        return {}

    summary = {}
    by_target = {}

    for entry in timeline:
        # 支持字典和 Pydantic 模型
        if hasattr(entry, 'predictions'):
            predictions = entry.predictions
        else:
            predictions = entry.get('predictions', {})

        for target, val in predictions.items():
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(val)

    for target, vals in by_target.items():
        arr = np.array(vals)
        summary[target] = {
            'label': TARGET_LABELS.get(target, target),
            'unit': TARGET_UNITS.get(target, ''),
            'min': round(float(arr.min()), 4),
            'max': round(float(arr.max()), 4),
            'mean': round(float(arr.mean()), 4),
            'std': round(float(arr.std()), 4),
        }

    return summary