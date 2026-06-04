#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA 模型（基于 pmdarima 自动定阶）

适用目标: 温度、湿度（有周期性）
可调参数:
  - seasonal: 是否考虑季节性 (True/False)
  - m: 周期长度 (96 = 15min×96 = 1天)
  - max_p/max_d/max_q: ARIMA 参数搜索范围
  - stepwise: 是否使用逐步搜索
  - information_criterion: 模型选择准则 (aic, bic)

注: ARIMA 是单变量模型，仅使用历史值预测未来
    超过 MAX_SIMPLE_SAMPLES 时禁用季节性以避免计算爆炸
    超过 MAX_ARIMA_SAMPLES 时截取最近 N 条
"""

import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

MAX_ARIMA_SAMPLES = 5000
MAX_SIMPLE_SAMPLES = 2000


class ARIMAModel:
    """ARIMA 模型包装器"""

    def __init__(self, **params):
        self.seasonal = params.get('seasonal', True)
        self.m = params.get('m', 96)
        self.max_p = params.get('max_p', 5)
        self.max_d = params.get('max_d', 2)
        self.max_q = params.get('max_q', 5)
        self.stepwise = params.get('stepwise', True)
        self.information_criterion = params.get('information_criterion', 'aic')
        self.model = None
        self.model_fit = None

    def fit(self, y_train):
        """训练 ARIMA 模型 (y_train: 1D array)"""
        import pmdarima as pm

        y = np.array(y_train, dtype=float)
        original_len = len(y)
        capped = False

        if original_len > MAX_ARIMA_SAMPLES:
            logger.warning(
                f"ARIMA 数据量 {original_len} 超过上限 {MAX_ARIMA_SAMPLES}，"
                f"截取最近 {MAX_ARIMA_SAMPLES} 条"
            )
            y = y[-MAX_ARIMA_SAMPLES:]
            capped = True

        use_seasonal = self.seasonal and not capped and len(y) <= MAX_SIMPLE_SAMPLES
        if self.seasonal and not use_seasonal:
            logger.info(f"ARIMA 数据量 {len(y)} 较大，禁用季节性以加速训练，m={self.m}")

        effective_m = min(self.m, 12) if use_seasonal else 1
        max_p = min(self.max_p, 2) if capped else self.max_p
        max_q = min(self.max_q, 2) if capped else self.max_q
        max_d = min(self.max_d, 1) if capped else self.max_d

        logger.info(
            f"ARIMA: seasonal={use_seasonal}, m={effective_m}, "
            f"max(p,d,q)=({max_p},{max_d},{max_q}), "
            f"samples={len(y)}"
        )

        try:
            self.model = pm.auto_arima(
                y,
                seasonal=use_seasonal,
                m=effective_m,
                start_p=0, max_p=max_p,
                start_d=0, max_d=max_d,
                start_q=0, max_q=max_q,
                max_P=1 if use_seasonal else 0,
                max_Q=1 if use_seasonal else 0,
                stepwise=True,
                information_criterion=self.information_criterion,
                suppress_warnings=True,
                error_action='ignore',
                trace=False,
                maxiter=30,
            )
        except Exception as e:
            logger.warning(f"auto_arima 失败: {e}, 降级为最简单模型")
            self.model = pm.auto_arima(
                y, seasonal=False,
                start_p=0, max_p=2, max_d=1, max_q=2,
                stepwise=True, suppress_warnings=True,
                error_action='ignore', trace=False,
            )

        self.model_fit = self.model
        logger.info(f"ARIMA 拟合完成: order={self.model.order}, seasonal_order={self.model.seasonal_order}")

    def predict(self, n_steps):
        """预测未来 n_steps 步"""
        if self.model is None:
            raise RuntimeError("模型未训练, 请先调用 fit()")
        return self.model.predict(n_periods=n_steps)

    def forecast(self, n_steps):
        """forecast 别名"""
        return self.predict(n_steps)


def train_arima(y_train, y_val, params=None, return_train_time=False):
    """
    训练 ARIMA 并评估

    Args:
        y_train: 训练序列 (1D)
        y_val:   验证序列 (1D)
        params:  参数字典

    Returns:
        (y_pred, metrics_dict) 或 (y_pred, metrics_dict, train_time)
    """
    from utils.metrics import mae, rmse, r2_score

    if params is None:
        params = {}

    t0 = time.time()
    model = ARIMAModel(**params)
    model.fit(y_train)
    y_pred = model.predict(len(y_val))
    train_time = time.time() - t0

    result = {
        'mae': mae(y_val, y_pred),
        'rmse': rmse(y_val, y_pred),
        'r2': r2_score(y_val, y_pred),
    }

    if return_train_time:
        return y_pred, result, train_time
    return y_pred, result