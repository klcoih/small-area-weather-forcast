#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
降雨量专项预测工具

1. 基础工具:
   - 二分类标签、分级、零膨胀率
   - 样本加权
   - log1p / expm1 变换

2. 两阶段预测:
   - 阶段1: XGBClassifier (有雨/无雨)
   - 阶段2: XGBRegressor (降雨量, 仅降雨样本)
   - 训练函数: train_rainfall_model
   - 预测函数: predict_rainfall
   - 评估函数: evaluate_rainfall

3. Markov 链辅助:
   - 状态离散化
   - 转移概率矩阵构建
   - Chapman-Kolmogorov 多步预测
"""

import time
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAIN_LABELS_4 = ['no_rain', 'light', 'moderate', 'heavy']


# ═══════════════════════════════════════════════════════════════
# 基础工具
# ═══════════════════════════════════════════════════════════════

def rain_binary_label(rainfall, threshold=0.1):
    """连续降雨量 → 二分类标签 (0=无雨, 1=有雨)"""
    return (np.array(rainfall, dtype=float) >= threshold).astype(int)


def rain_categories(rainfall):
    """降雨量分级: 0=无雨, 1=小雨, 2=中雨, 3=大雨"""
    r = np.array(rainfall, dtype=float)
    cats = np.zeros(len(r), dtype=int)
    cats[r >= 0.1] = 1
    cats[r >= 0.625] = 2
    cats[r >= 1.875] = 3
    return cats


def rain_zero_inflation_ratio(rainfall, threshold=0.1):
    """零膨胀比例"""
    return float(np.mean(np.array(rainfall) < threshold))


def rain_sample_weight(rainfall, base_weight=1.0):
    """非零降雨样本加权"""
    r = np.array(rainfall, dtype=float)
    weights = np.full(len(r), base_weight, dtype=float)
    rain_mask = r >= 0.1
    weights[rain_mask] = base_weight + r[rain_mask]
    return weights


def split_rain_data(X, rainfall, threshold=0.1):
    """将数据分为无雨子集和有雨子集"""
    rain_flag = rain_binary_label(rainfall, threshold)
    mask_rain = rain_flag == 1
    if isinstance(X, np.ndarray):
        X_rain = X[mask_rain]
        X_no_rain = X[~mask_rain]
    else:
        X_rain = X.iloc[mask_rain]
        X_no_rain = X.iloc[~mask_rain]
    r = np.array(rainfall, dtype=float)
    return X_no_rain, X_rain, r[~mask_rain], r[mask_rain], mask_rain


def log1p_transform(values):
    """log(1+x) 变换，缓解右偏分布"""
    return np.log1p(np.array(values, dtype=float))


def expm1_inverse(values):
    """exp(x)-1 逆变换"""
    return np.expm1(np.array(values, dtype=float))


def compute_scale_pos_weight(y_binary):
    """计算类别不平衡权重: neg_count / pos_count"""
    y = np.array(y_binary, dtype=int)
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


# ═══════════════════════════════════════════════════════════════
# 特征工程
# ═══════════════════════════════════════════════════════════════

def generate_rainfall_features(df, rainfall_col='rainfall'):
    """
    生成降雨专用特征

    基于现有数据基础上添加:
      - rain_rate: 降雨率 (当前/前4步的比值)
      - consecutive_dry: 连续无雨步数
      - rain_intensity: 降雨强度 (当前/滚动累计)

    Args:
        df: DataFrame, 需含 rainfall 列

    Returns:
        DataFrame: 添加了降雨专用特征
    """
    df = df.copy()
    r = df[rainfall_col].values

    shifted_sum = pd.Series(r).rolling(4, min_periods=1).sum().values
    df['rain_intensity'] = np.divide(r, shifted_sum + 1e-6, dtype=float)

    dry_flag = (r < 0.1).astype(int)
    consecutive = np.zeros(len(r), dtype=int)
    count = 0
    for i in range(len(r)):
        if dry_flag[i]:
            count += 1
        else:
            count = 0
        consecutive[i] = count
    df['consecutive_dry'] = consecutive

    logger.info("降雨专用特征: rain_intensity, consecutive_dry")
    return df


# ═══════════════════════════════════════════════════════════════
# 两阶段模型 — 训练
# ═══════════════════════════════════════════════════════════════

def train_rainfall_model(X_train, y_train_rainfall, X_val, y_val_rainfall,
                          threshold=0.1, use_log1p=True,
                          cls_params=None, reg_params=None):
    """
    降雨量两阶段预测模型

    阶段1 (分类): 预测有雨/无雨 (rain_flag)
    阶段2 (回归):  仅降雨样本预测降雨量

    Args:
        X_train, y_train_rainfall: 训练数据
        X_val, y_val_rainfall:     验证数据
        threshold:                  降雨阈值 (mm)
        use_log1p:                  是否对降雨量做 log1p 变换
        cls_params:                 分类器参数
        reg_params:                 回归器参数

    Returns:
        ((classifier, regressor), metrics_dict, train_time)
    """
    from models.xgboost_model import train_xgboost_rainfall_two_stage

    yr_train = np.array(y_train_rainfall, dtype=float)
    yr_val = np.array(y_val_rainfall, dtype=float)

    y_train_flag = rain_binary_label(yr_train, threshold)
    y_val_flag = rain_binary_label(yr_val, threshold)

    if use_log1p:
        rain_mask_train = y_train_flag == 1
        yr_train_transformed = yr_train.copy()
        yr_train_transformed[rain_mask_train] = log1p_transform(yr_train[rain_mask_train])

        rain_mask_val = y_val_flag == 1
        yr_val_transformed = yr_val.copy()
        yr_val_transformed[rain_mask_val] = log1p_transform(yr_val[rain_mask_val])
    else:
        yr_train_transformed = yr_train
        yr_val_transformed = yr_val

    params = {}
    if cls_params:
        params['classification'] = cls_params
    if reg_params:
        params['regression'] = reg_params

    return train_xgboost_rainfall_two_stage(
        X_train, y_train_flag, yr_train_transformed,
        X_val, y_val_flag, yr_val_transformed, params
    )


def train_rainfall_markov(y_train, y_val, n_states=2, thresholds=None, n_steps=1):
    """
    Markov 链降雨状态预测

    Args:
        y_train: 训练降雨量序列
        y_val:   验证降雨量序列
        n_states: 状态数
        thresholds: 离散化阈值
        n_steps: 预测步数

    Returns:
        (pred_states, metrics, train_time)
    """
    from models.markov_model import train_markov

    params = {
        'n_states': n_states,
        'thresholds': thresholds or [0.1] if n_states == 2 else [0.1, 0.625, 1.875],
        'n_steps': n_steps,
    }
    return train_markov(np.array(y_train), np.array(y_val), params)


# ═══════════════════════════════════════════════════════════════
# 两阶段模型 — 预测
# ═══════════════════════════════════════════════════════════════

def predict_rainfall(X, classifier, regressor, use_log1p=True, threshold=0.5):
    """
    两阶段降雨量预测

    Args:
        X:           输入特征
        classifier:  阶段1 分类器 (XGBClassifier)
        regressor:   阶段2 回归器 (XGBRegressor)
        use_log1p:   是否对预测值做 expm1 逆变换
        threshold:   分类概率阈值

    Returns:
        (rain_flag, rainfall):
          rain_flag: 0/1 有雨标志
          rainfall:  预测降雨量 (mm)
    """
    try:
        rain_prob = classifier.predict_proba(X)[:, 1]
    except AttributeError:
        rain_prob = classifier.predict(X).flatten()

    rain_flag = (rain_prob >= threshold).astype(int)
    rainfall = np.zeros(len(X))

    rain_indices = np.where(rain_flag == 1)[0]
    if len(rain_indices) > 0:
        if isinstance(X, pd.DataFrame):
            X_rain = X.iloc[rain_indices]
        else:
            X_rain = X[rain_indices]

        rain_pred = regressor.predict(X_rain)
        if use_log1p:
            rain_pred = expm1_inverse(rain_pred)
        rainfall[rain_indices] = np.maximum(rain_pred, 0.0)

    return rain_flag, rainfall


# ═══════════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════════

def evaluate_rainfall(y_true_rainfall, y_pred_flag, y_pred_rainfall,
                       threshold=0.1, y_pred_prob=None):
    """
    降雨量完整评估

    Args:
        y_true_rainfall: 真实降雨量
        y_pred_flag:     预测的有雨标志 (0/1)
        y_pred_rainfall: 预测的降雨量
        threshold:       降雨阈值
        y_pred_prob:     分类概率 (可选，用于 AUC)

    Returns:
        dict: classification + regression 指标
    """
    from utils.metrics import auc, brier_score, csi, mae, rmse

    yt = np.array(y_true_rainfall, dtype=float).flatten()
    yp_flag = np.array(y_pred_flag, dtype=int).flatten()
    yp_rain = np.array(y_pred_rainfall, dtype=float).flatten()

    yt_flag = rain_binary_label(yt, threshold)

    cls_metrics = {
        'csi': csi(yt_flag, yp_flag),
        'accuracy': float(np.mean(yt_flag == yp_flag)),
    }

    if y_pred_prob is not None:
        yp_prob = np.array(y_pred_prob, dtype=float).flatten()
        cls_metrics['auc'] = auc(yt_flag, yp_prob)
        cls_metrics['brier'] = brier_score(yt_flag, yp_prob)

    rain_mask = yt_flag == 1
    if rain_mask.sum() > 0:
        reg_metrics = {
            'mae_rain_only': mae(yt[rain_mask], yp_rain[rain_mask]),
            'rmse_rain_only': rmse(yt[rain_mask], yp_rain[rain_mask]),
        }
    else:
        reg_metrics = {'mae_rain_only': 0.0, 'rmse_rain_only': 0.0}

    reg_metrics['mae_all'] = mae(yt, yp_rain)

    return {
        'classification': cls_metrics,
        'regression': reg_metrics,
    }


# ═══════════════════════════════════════════════════════════════
# 综合训练接口 (兼容 train.py)
# ═══════════════════════════════════════════════════════════════

def train_rainfall(X_train, y_train, X_val, y_val,
                    model_type='xgboost', params=None,
                    threshold=0.1, use_log1p=True):
    """
    降雨量预测统一入口 (train.py 兼容)

    Args:
        X_train, y_train: 训练数据
        X_val, y_val:     验证数据
        model_type:       xgboost / markov / lstm
        params:           模型参数
        threshold:        降雨阈值
        use_log1p:        是否 log1p 变换

    Returns:
        ((y_pred_flag, y_pred_rainfall), metrics_dict, train_time)
        或 (pred_states, metrics_dict, train_time)  for markov
    """
    if params is None:
        params = {}

    if model_type == 'xgboost':
        (y_prob, y_pred_rain), metrics, t = train_rainfall_model(
            X_train, y_train, X_val, y_val,
            threshold=threshold, use_log1p=use_log1p,
            cls_params=params.get('classification'),
            reg_params=params.get('regression'),
        )
        return (y_prob, y_pred_rain), metrics, t

    elif model_type == 'markov':
        return train_rainfall_markov(
            y_train, y_val,
            n_states=params.get('n_states', 2),
            thresholds=params.get('thresholds', [threshold]),
            n_steps=params.get('n_steps', 1),
        )

    elif model_type == 'lstm':
        from models.lstm_model import train_lstm_rainfall_two_stage

        yr_train = np.array(y_train, dtype=float)
        yr_val = np.array(y_val, dtype=float)
        yt_flag = rain_binary_label(yr_train, threshold)
        yv_flag = rain_binary_label(yr_val, threshold)

        if use_log1p:
            mask_tr = yt_flag == 1
            mask_vl = yv_flag == 1
            yr_train_t = yr_train.copy()
            yr_val_t = yr_val.copy()
            yr_train_t[mask_tr] = log1p_transform(yr_train[mask_tr])
            yr_val_t[mask_vl] = log1p_transform(yr_val[mask_vl])
        else:
            yr_train_t = yr_train
            yr_val_t = yr_val

        return train_lstm_rainfall_two_stage(
            X_train, yt_flag, yr_train_t,
            X_val, yv_flag, yr_val_t, params
        )

    else:
        raise ValueError(f"降雨不支持模型: {model_type}")