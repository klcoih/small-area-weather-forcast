#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""特殊预处理工具：风向、降雨、序列转换"""

import numpy as np
import pandas as pd


def prepare_for_arima(series, train_ratio=0.8):
    """将时间序列拆分为 ARIMA 所需的 train/test"""
    n = len(series)
    split = int(n * train_ratio)
    return series.iloc[:split].values, series.iloc[split:].values


def prepare_for_lstm(X, y, seq_length=96):
    """
    将表格数据转为 LSTM 序列格式 (samples, seq_length, features)
    X: DataFrame or ndarray, shape (n_samples, n_features)
    y: Series or ndarray, shape (n_samples,)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values.astype(np.float32)
    else:
        X = np.array(X, dtype=np.float32)
    if isinstance(y, pd.Series):
        y = y.values.astype(np.float32)
    else:
        y = np.array(y, dtype=np.float32)

    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def prepare_sequences_lstm(X, y, seq_length):
    """prepare_for_lstm 的别名"""
    return prepare_for_lstm(X, y, seq_length)


def prepare_for_prophet(df, timestamp_col='timestamp', target_col='y'):
    """转为 Prophet 所需的两列格式 ds, y"""
    df_p = pd.DataFrame({
        'ds': pd.to_datetime(df[timestamp_col]),
        'y': df[target_col].values
    })
    return df_p


def prepare_for_markov_discretize(series, thresholds):
    """
    将连续值离散化为 Markov 状态

    Args:
        series: 1D array
        thresholds: list of (min, max) tuples or list of boundary values

    Returns:
        states: 1D integer array (0, 1, 2, ...)
        n_states: number of states
    """
    values = np.array(series, dtype=float)
    boundaries = sorted(thresholds)
    states = np.zeros(len(values), dtype=int)
    for i, b in enumerate(boundaries):
        states[values >= b] = i + 1
    return states, len(boundaries) + 1


def prepare_for_markov(series, thresholds):
    """prepare_for_markov_discretize 的别名"""
    return prepare_for_markov_discretize(series, thresholds)


def normalize_data(X, y=None, scaler=None):
    """使用均值-标准差归一化"""
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.array(X, dtype=np.float32)
    if scaler is None:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0) + 1e-8
        scaler = (mean, std)
    mean, std = scaler
    X_norm = (X - mean) / std
    if y is not None:
        y = np.array(y, dtype=np.float32)
        y_mean = np.nanmean(y)
        y_std = np.nanstd(y) + 1e-8
        y_norm = (y - y_mean) / y_std
        return X_norm, y_norm, (scaler, (y_mean, y_std))
    return X_norm, scaler


def denormalize_y(y_norm, y_scaler):
    """反归一化预测值"""
    y_mean, y_std = y_scaler
    return y_norm * y_std + y_mean