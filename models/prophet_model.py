#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prophet 模型（Facebook Prophet）

适用目标: 温度、湿度、光照强度（有明显日/年周期）

可调参数:
  - changepoint_prior_scale: 趋势变化灵活性 (0.001-0.5)
  - seasonality_prior_scale: 季节性强度 (0.01-10)
  - yearly_seasonality: 年周期性 (True/False)
  - daily_seasonality: 日周期性 (True/False)
    (15分钟数据需设置 daily_seasonality=True)
  - interval_width: 预测区间宽度 (0.8)

注: Prophet 也是单变量模型，仅使用时间和历史值
     超过 max_samples 时自动截取
"""

import time
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

MAX_PROPHET_SAMPLES = 20000


class ProphetModel:
    """Prophet 模型包装器"""

    def __init__(self, **params):
        self.changepoint_prior_scale = params.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = params.get('seasonality_prior_scale', 10.0)
        self.yearly_seasonality = params.get('yearly_seasonality', True)
        self.daily_seasonality = params.get('daily_seasonality', True)
        self.interval_width = params.get('interval_width', 0.8)
        self.model = None

    def fit(self, df_train):
        """
        训练 Prophet 模型

        Args:
            df_train: DataFrame with columns ['ds', 'y']
                      ds: datetime, y: float
        """
        from prophet import Prophet

        original_len = len(df_train)
        if original_len > MAX_PROPHET_SAMPLES:
            logger.warning(
                f"Prophet 数据量 {original_len} 超过上限 {MAX_PROPHET_SAMPLES}，"
                f"截取最近 {MAX_PROPHET_SAMPLES} 条"
            )
            df_train = df_train.iloc[-MAX_PROPHET_SAMPLES:].copy()

        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width,
        )
        self.model.fit(df_train)
        logger.info(
            f"Prophet: changepoint={self.changepoint_prior_scale}, "
            f"seasonality={self.seasonality_prior_scale}, "
            f"yearly={self.yearly_seasonality}, daily={self.daily_seasonality}"
        )

    def predict(self, periods, freq='15min'):
        """
        预测未来

        Args:
            periods: 预测步数
            freq: 频率字符串

        Returns:
            DataFrame: 完整预测结果 (ds, yhat, yhat_lower, yhat_upper)
        """
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast


def train_prophet(y_train_series, y_val_series, timestamps_train=None,
                  timestamps_val=None, params=None):
    """
    训练 Prophet 并评估

    Args:
        y_train_series: 训练序列 (pd.Series with datetime index) 或 (pd.Series, pd.Series) with timestamps
        y_val_series:   验证序列
        timestamps_train: 训练时间戳 (datetime)
        timestamps_val:   验证时间戳 (datetime)
        params:          参数字典

    Returns:
        (y_pred, metrics, train_time)
    """
    from utils.metrics import mae, rmse, r2_score

    if params is None:
        params = {}

    if timestamps_train is not None:
        df_train = pd.DataFrame({
            'ds': pd.to_datetime(timestamps_train),
            'y': np.array(y_train_series, dtype=float)
        })
    elif hasattr(y_train_series, 'index') and isinstance(y_train_series.index, pd.DatetimeIndex):
        df_train = pd.DataFrame({
            'ds': y_train_series.index,
            'y': y_train_series.values
        })
    else:
        df_train = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=len(y_train_series), freq='15min'),
            'y': np.array(y_train_series, dtype=float)
        })

    y_val = np.array(y_val_series, dtype=float)

    t0 = time.time()
    model = ProphetModel(**params)
    model.fit(df_train)
    forecast = model.predict(len(y_val), freq='15min')
    y_pred = forecast['yhat'].values[-len(y_val):]
    train_time = time.time() - t0

    return y_pred, {
        'mae': mae(y_val, y_pred),
        'rmse': rmse(y_val, y_pred),
        'r2': r2_score(y_val, y_pred),
    }, train_time