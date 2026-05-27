#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风速 & 风向专项预测工具

1. 风速:
   - 加权 MSE (大风权重=1+风速)
   - 风速特征工程
   - 训练函数 (XGBoost/LSTM)
   - 评估函数

2. 风向:
   - 角度↔向量转换 (sin, cos)
   - 环形 MSE 损失
   - 静风处理 (风速<阈值时剔除)
   - 训练函数 (LSTM/XGBoost)
   - 评估函数 (角度误差)
"""

import time
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DIRECTION_LABELS_8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
DIRECTION_LABELS_16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                        'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

# ═══════════════════════════════════════════════════════════════
# 角度 ↔ 向量 转换
# ═══════════════════════════════════════════════════════════════

def angle_to_components(angle_deg):
    """角度(0-360°) → (sin, cos)"""
    rad = np.radians(np.array(angle_deg, dtype=float))
    return np.sin(rad), np.cos(rad)


def components_to_angle(sin, cos):
    """(sin, cos) → 角度(0-360°)"""
    angle = np.degrees(np.arctan2(np.array(sin), np.array(cos)))
    return angle % 360


def angular_difference(a, b):
    """两个角度的最小绝对差 (0-180)"""
    diff = np.abs(np.array(a) - np.array(b)) % 360
    return np.minimum(diff, 360 - diff)


def angular_mean(angles):
    """角度的循环均值"""
    rad = np.radians(np.array(angles))
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360)


def angular_std(angles):
    """角度的循环标准差"""
    rad = np.radians(np.array(angles))
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    R = np.sqrt(mean_sin ** 2 + mean_cos ** 2)
    return float(np.degrees(np.sqrt(-2 * np.log(R))) if R > 0 else 180.0)


def discretize_wind_direction(angle_deg, n_directions=8):
    """将风向离散化为 N 个方向"""
    angle = np.array(angle_deg) % 360
    bin_width = 360.0 / n_directions
    return (angle / bin_width).astype(int) % n_directions


# ═══════════════════════════════════════════════════════════════
# 风速 —— 加权损失 & 特征工程
# ═══════════════════════════════════════════════════════════════

def weighted_mse(y_true, y_pred):
    """
    风速加权 MSE: 大风权重 = 1 + wind_speed

    Args:
        y_true: 真实风速
        y_pred: 预测风速

    Returns:
        float: 加权均方误差
    """
    yt = np.array(y_true, dtype=float).flatten()
    yp = np.array(y_pred, dtype=float).flatten()
    weights = 1.0 + yt
    return float(np.mean(weights * (yt - yp) ** 2))


def weighted_mae(y_true, y_pred):
    """风速加权 MAE"""
    yt = np.array(y_true, dtype=float).flatten()
    yp = np.array(y_pred, dtype=float).flatten()
    weights = 1.0 + yt
    return float(np.mean(weights * np.abs(yt - yp)))


def generate_wind_speed_features(df):
    """
    生成风速专用特征

    基于现有数据的滞后特征基础上，添加:
      - wind_gust_idx: 阵风指数 (当前/滚动最大)
      - wind_trend: 风速趋势 (当前-4步前)

    Args:
        df: DataFrame, 需含 'wind_speed' 列

    Returns:
        DataFrame: 添加了风速专用特征
    """
    df = df.copy()
    ws = df['wind_speed'].values

    rolling_max_4 = pd.Series(ws).rolling(4, min_periods=1).max().values
    df['wind_gust_idx'] = np.divide(ws, rolling_max_4 + 1e-6, dtype=float)

    df['wind_trend'] = ws - pd.Series(ws).shift(4).fillna(method='bfill').values

    logger.info("风速专用特征: wind_gust_idx, wind_trend")
    return df


def train_wind_speed_model(X_train, y_train, X_val, y_val, model_type='xgboost', params=None):
    """
    风速预测模型训练

    Args:
        X_train, y_train: 训练数据
        X_val, y_val:     验证数据
        model_type:       xgboost / lstm
        params:           模型参数

    Returns:
        (y_pred, metrics_dict, train_time)
    """
    if params is None:
        params = {}

    yt = np.array(y_train, dtype=float)
    yv = np.array(y_val, dtype=float)

    if model_type == 'xgboost':
        from models.xgboost_model import train_xgboost_wind_speed_weighted
        return train_xgboost_wind_speed_weighted(X_train, yt, X_val, yv, params)

    elif model_type == 'lstm':
        from models.lstm_model import train_lstm_regression
        from utils.metrics import mae, rmse, r2_score

        lstm_params = dict(params)
        lstm_params.setdefault('seq_length', 96)
        lstm_params.setdefault('hidden_size', 64)
        lstm_params.setdefault('num_layers', 2)
        lstm_params.setdefault('dropout', 0.2)
        lstm_params.setdefault('learning_rate', 0.001)
        lstm_params.setdefault('batch_size', 64)
        lstm_params.setdefault('epochs', 30)

        y_pred, metrics, t = train_lstm_regression(X_train, yt, X_val, yv, lstm_params)
        metrics['weighted_mae'] = weighted_mae(yv[:len(y_pred)], y_pred)
        metrics['weighted_mse'] = weighted_mse(yv[:len(y_pred)], y_pred)
        return y_pred, metrics, t

    else:
        raise ValueError(f"风速不支持模型: {model_type}")


def evaluate_wind_speed(y_true, y_pred):
    """风速评估：标准指标 + 加权指标 + 大风事件召回"""
    from utils.metrics import mae, rmse, r2_score

    yt = np.array(y_true, dtype=float).flatten()
    yp = np.array(y_pred, dtype=float).flatten()

    high_wind_mask = yt >= np.percentile(yt, 90)
    high_wind_mae = mae(yt[high_wind_mask], yp[high_wind_mask]) if high_wind_mask.sum() > 0 else 0.0

    return {
        'mae': mae(yt, yp),
        'rmse': rmse(yt, yp),
        'r2': r2_score(yt, yp),
        'weighted_mae': weighted_mae(yt, yp),
        'weighted_mse': weighted_mse(yt, yp),
        'high_wind_mae': round(high_wind_mae, 4),
    }


# ═══════════════════════════════════════════════════════════════
# 风向 —— 环形损失 & 静风处理
# ═══════════════════════════════════════════════════════════════

def circular_mse(y_true, y_pred):
    """
    环形 MSE 损失（使用余弦相似度）

    Args:
        y_true: (N, 2) [sin, cos]
        y_pred: (N, 2) [sin, cos]

    Returns:
        float: 环形均方误差
    """
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 2)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 2)

    dot_product = np.sum(yt * yp, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_diff = np.arccos(dot_product)
    return float(np.mean(angle_diff ** 2))


def circular_mae(y_true, y_pred):
    """环形 MAE (弧度)"""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 2)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 2)
    dot_product = np.sum(yt * yp, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_diff = np.arccos(dot_product)
    return float(np.mean(angle_diff))


def filter_calm_wind(df, wind_speed_col='wind_speed', threshold=0.5):
    """
    静风过滤: 风速 < threshold 时风向无实际意义

    Returns:
        (filtered_df, calm_mask)
    """
    if isinstance(df, pd.DataFrame):
        calm_mask = df[wind_speed_col] < threshold
        calm_count = int(calm_mask.sum())
        logger.info(f"静风过滤: {calm_count} 条 (风速<{threshold}m/s), 占比 {calm_count/len(df):.1%}")
        return df[~calm_mask].copy(), calm_mask
    else:
        ws = np.array(df) if np.ndim(df) == 1 else df
        calm_mask = ws < threshold
        return df[~calm_mask], calm_mask


def prepare_wind_direction_data(df, wind_dir_col='wind_direction',
                                 wind_speed_col='wind_speed',
                                 calm_threshold=0.5,
                                 filter_calm=True):
    """
    准备风向训练数据: 角度→向量 + 可选静风过滤

    Returns:
        X_train, y_sin, y_cos (或带过滤后的完整 DataFrame)
    """
    df = df.copy()

    if filter_calm and wind_speed_col in df.columns:
        df, calm_mask = filter_calm_wind(df, wind_speed_col, calm_threshold)

    angle = df[wind_dir_col].values
    sin_vals, cos_vals = angle_to_components(angle)
    df['wind_sin'] = sin_vals
    df['wind_cos'] = cos_vals

    return df


def train_wind_direction_model(X_train, y_train_angle, X_val, y_val_angle,
                                model_type='lstm', params=None,
                                wind_speed_train=None, wind_speed_val=None,
                                calm_threshold=0.5):
    """
    风向预测模型训练

    Args:
        X_train, y_train_angle: 训练特征和目标角度
        X_val, y_val_angle:     验证特征和目标角度
        model_type:             lstm / xgboost
        params:                 模型参数
        wind_speed_train/val:   用于静风过滤的风速序列
        calm_threshold:         静风阈值

    Returns:
        (y_pred_components, metrics_dict, train_time)
    """
    if params is None:
        params = {}

    y_train = np.array(y_train_angle, dtype=float)
    y_val = np.array(y_val_angle, dtype=float)

    if wind_speed_train is not None:
        ws_train = np.array(wind_speed_train, dtype=float)
        ws_val = np.array(wind_speed_val, dtype=float)
        train_mask = ws_train >= calm_threshold
        val_mask = ws_val >= calm_threshold

        if isinstance(X_train, pd.DataFrame):
            X_train_f = X_train.iloc[train_mask].reset_index(drop=True)
            X_val_f = X_val.iloc[val_mask].reset_index(drop=True)
        else:
            X_train_f = X_train[train_mask]
            X_val_f = X_val[val_mask]
        y_train_f = y_train[train_mask]
        y_val_f = y_val[val_mask]
        logger.info(f"静风过滤: train {len(X_train_f)}/{len(X_train)}, val {len(X_val_f)}/{len(X_val)}")
    else:
        X_train_f, X_val_f = X_train, X_val
        y_train_f, y_val_f = y_train, y_val

    y_train_sin, y_train_cos = angle_to_components(y_train_f)
    y_val_sin, y_val_cos = angle_to_components(y_val_f)

    if model_type == 'xgboost':
        from models.xgboost_model import train_xgboost_wind
        return train_xgboost_wind(X_train_f, y_train_sin, y_train_cos,
                                  X_val_f, y_val_sin, y_val_cos, params)

    elif model_type == 'lstm':
        from models.lstm_model import train_lstm_wind
        return train_lstm_wind(X_train_f, y_train_sin, y_train_cos,
                               X_val_f, y_val_sin, y_val_cos, params)

    else:
        raise ValueError(f"风向不支持模型: {model_type}")


def evaluate_wind_direction(y_true_angle, y_pred_sin, y_pred_cos):
    """
    风向评估: 角度误差 + 环形损失

    Args:
        y_true_angle: 真实角度 (0-360)
        y_pred_sin/y_pred_cos: 预测的 sin/cos 分量

    Returns:
        dict: angular_error, circular_mse, circular_mae, accuracy_22_5, accuracy_45
    """
    yt = np.array(y_true_angle, dtype=float).flatten()
    yp_sin = np.array(y_pred_sin, dtype=float).flatten()
    yp_cos = np.array(y_pred_cos, dtype=float).flatten()

    yt_sin, yt_cos = angle_to_components(yt)
    y_pred_angle = components_to_angle(yp_sin, yp_cos)

    ae = angular_difference(yt, y_pred_angle)
    mean_ae = float(np.mean(ae))

    circ_mse = circular_mse(
        np.column_stack([yt_sin, yt_cos]),
        np.column_stack([yp_sin, yp_cos])
    )

    acc_22_5 = float(np.mean(ae <= 22.5))
    acc_45 = float(np.mean(ae <= 45.0))

    return {
        'angular_error': round(mean_ae, 3),
        'circular_mse': round(circ_mse, 6),
        'circular_mae': round(circular_mae(
            np.column_stack([yt_sin, yt_cos]),
            np.column_stack([yp_sin, yp_cos])
        ), 4),
        'accuracy_22_5deg': round(acc_22_5, 4),
        'accuracy_45deg': round(acc_45, 4),
    }


# ═══════════════════════════════════════════════════════════════
# 综合
# ═══════════════════════════════════════════════════════════════

def generate_wind_features(df, wind_speed_col='wind_speed',
                           wind_dir_col='wind_direction'):
    """
    一站式生成所有风速/风向专用特征
    """
    df = df.copy()
    df = generate_wind_speed_features(df)
    if wind_dir_col in df.columns:
        sin_vals, cos_vals = angle_to_components(df[wind_dir_col].values)
        df['wind_sin'] = sin_vals
        df['wind_cos'] = cos_vals
    return df