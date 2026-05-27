#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 模型（XGBRegressor / XGBClassifier）

适用目标: 所有目标（特征丰富时最佳）
可调参数:
  - max_depth: 树深度 (3-10)
  - learning_rate: 学习率 (0.01-0.3)
  - n_estimators: 树数量 (100-1000)
  - subsample: 样本采样比例 (0.6-1.0)
  - colsample_bytree: 特征采样比例 (0.6-1.0)
  - reg_alpha/reg_lambda: 正则化
  - min_child_weight: 最小叶子权重

特殊:
  - 风速: 自定义加权损失 (大风权重=1+风速)
  - 降雨: 两阶段 (XGBClassifier + XGBRegressor)
"""

import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


class XGBoostRegressor:
    """XGBoost 回归器"""

    def __init__(self, **params):
        self.max_depth = params.get('max_depth', 6)
        self.learning_rate = params.get('learning_rate', 0.05)
        self.n_estimators = params.get('n_estimators', 300)
        self.subsample = params.get('subsample', 0.8)
        self.colsample_bytree = params.get('colsample_bytree', 0.8)
        self.reg_alpha = params.get('reg_alpha', 0)
        self.reg_lambda = params.get('reg_lambda', 1)
        self.min_child_weight = params.get('min_child_weight', 1)
        self.model = None

    def fit(self, X_train, y_train, sample_weight=None):
        import xgboost as xgb

        self.model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)


class XGBoostClassifier:
    """XGBoost 分类器"""

    def __init__(self, **params):
        self.max_depth = params.get('max_depth', 5)
        self.learning_rate = params.get('learning_rate', 0.1)
        self.n_estimators = params.get('n_estimators', 200)
        self.subsample = params.get('subsample', 0.8)
        self.model = None

    def fit(self, X_train, y_train):
        import xgboost as xgb

        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            objective='binary:logistic',
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def train_xgboost_regression(X_train, y_train, X_val, y_val,
                             params=None, sample_weight=None):
    """
    训练 XGBoost 回归

    Returns:
        (y_pred, metrics, train_time)
    """
    from utils.metrics import mae, rmse, r2_score

    if params is None:
        params = {}

    t0 = time.time()
    model = XGBoostRegressor(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_val)
    train_time = time.time() - t0

    return y_pred, {
        'mae': mae(y_val, y_pred),
        'rmse': rmse(y_val, y_pred),
        'r2': r2_score(y_val, y_pred),
    }, train_time


def train_xgboost_wind(X_train, y_train_sin, y_train_cos,
                       X_val, y_val_sin, y_val_cos, params=None):
    """
    XGBoost 风向预测 (输出 sin/cos)
    """
    from utils.metrics import angular_error_from_components

    if params is None:
        params = {}

    t0 = time.time()
    model_sin = XGBoostRegressor(**params)
    model_cos = XGBoostRegressor(**params)
    model_sin.fit(X_train, y_train_sin)
    model_cos.fit(X_train, y_train_cos)

    sin_pred = model_sin.predict(X_val)
    cos_pred = model_cos.predict(X_val)
    train_time = time.time() - t0

    angular_err = angular_error_from_components(y_val_sin, y_val_cos, sin_pred, cos_pred)

    return np.column_stack([sin_pred, cos_pred]), {
        'angular_error': angular_err,
    }, train_time


def train_xgboost_rainfall_two_stage(X_train, y_train_flag, y_train_rain,
                                     X_val, y_val_flag, y_val_rain, params=None):
    """
    XGBoost 降雨两阶段预测

    阶段1: 分类 (rain/no-rain)
    阶段2: 回归 (降雨量, 仅降雨样本)

    Returns:
        ((y_prob, y_pred_rain), metrics, train_time)
    """
    from utils.metrics import auc, brier_score, csi, mae

    if params is None:
        params = {}
    cls_params = params.get('classification', {
        'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200, 'subsample': 0.8
    })
    reg_params = params.get('regression', {
        'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8
    })

    t0 = time.time()

    cls_model = XGBoostClassifier(**cls_params)
    cls_model.fit(X_train, y_train_flag)
    y_prob = cls_model.predict_proba(X_val)
    y_pred_binary = (y_prob >= 0.5).astype(int)

    cls_metrics = {
        'auc': auc(y_val_flag, y_prob),
        'brier': brier_score(y_val_flag, y_prob),
        'csi': csi(y_val_flag, y_pred_binary),
    }

    train_mask = np.array(y_train_flag) == 1
    val_mask = np.array(y_val_flag) == 1

    if train_mask.sum() > 5 and val_mask.sum() > 0:
        reg_model = XGBoostRegressor(**reg_params)
        if isinstance(X_train, np.ndarray):
            reg_model.fit(X_train[train_mask], np.array(y_train_rain)[train_mask])
            y_pred_rain = np.zeros(len(X_val))
            y_pred_rain[val_mask] = reg_model.predict(X_val[val_mask])
        else:
            reg_model.fit(X_train.iloc[train_mask], np.array(y_train_rain)[train_mask])
            y_pred_rain = np.zeros(len(X_val))
            y_pred_rain[val_mask] = reg_model.predict(X_val.iloc[val_mask])
        rain_mae = mae(np.array(y_val_rain)[val_mask], y_pred_rain[val_mask])
    else:
        y_pred_rain = np.zeros(len(X_val))
        rain_mae = 0.0

    reg_metrics = {'mae_rain_only': rain_mae}
    train_time = time.time() - t0

    return (y_prob, y_pred_rain), {
        'classification': cls_metrics,
        'regression': reg_metrics,
    }, train_time


def train_xgboost_wind_speed_weighted(X_train, y_train, X_val, y_val, params=None):
    """
    XGBoost 风速预测 (大风权重=1+风速)
    """
    if params is None:
        params = {}

    yt = np.array(y_train, dtype=float)
    sample_weight = 1.0 + yt

    return train_xgboost_regression(X_train, y_train, X_val, y_val,
                                    params, sample_weight=sample_weight)