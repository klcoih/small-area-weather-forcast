#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练并保存最佳 XGBoost 模型到 saved_models/

用法:
  python save_models.py                  # 训练全部8个目标
  python save_models.py --target outdoor_temperature  # 只训练指定目标
"""

import os
import sys
import json
import time
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, '..'))
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'autoresearch_data')
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models')

TARGET_TYPES = {
    'greenhouse_temperature': 'regression',
    'greenhouse_humidity': 'regression',
    'outdoor_temperature': 'regression',
    'outdoor_humidity': 'regression',
    'light_intensity': 'regression',
    'wind_direction': 'angular_regression',
    'wind_speed': 'regression',
    'rainfall': 'two_stage',
}

TARGET_LABELS = {
    'greenhouse_temperature': '大棚温度 (°C)',
    'greenhouse_humidity': '大棚湿度 (%)',
    'outdoor_temperature': '室外温度 (°C)',
    'outdoor_humidity': '室外湿度 (%)',
    'light_intensity': '光照强度 (lux)',
    'wind_direction': '风向 (deg)',
    'wind_speed': '风速 (m/s)',
    'rainfall': '降雨量 (mm)',
}

from configs.preprocessing_config import PREPROCESS_CONFIG

SPECIAL_CONFIG = PREPROCESS_CONFIG.get('special_preprocessors', {})
FEATURE_CONFIG = PREPROCESS_CONFIG['features']
CROSS_CONFIG = PREPROCESS_CONFIG.get('cross_features', {})

GREENHOUSE_TARGETS = [t for t in TARGET_TYPES if TARGET_TYPES[t] == 'regression'
                       and t.startswith('greenhouse')]
OUTDOOR_REGRESSION_TARGETS = [t for t in TARGET_TYPES if TARGET_TYPES[t] == 'regression'
                              and not t.startswith('greenhouse')]


def load_data(target_name):
    target_dir = os.path.join(DATA_DIR, target_name)
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"数据目录不存在: {target_dir}")

    X_train = pd.read_csv(os.path.join(target_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(target_dir, 'y_train.csv'))['y']
    X_val = pd.read_csv(os.path.join(target_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(target_dir, 'y_val.csv'))['y']
    X_test = pd.read_csv(os.path.join(target_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(target_dir, 'y_test.csv'))['y']

    X_all = pd.concat([X_train, X_val, X_test], ignore_index=True)
    y_all = pd.concat([y_train, y_val, y_test], ignore_index=True)

    logger.info(f"加载 {target_name}: total={X_all.shape}, features={X_all.shape[1]}")
    return X_all, y_all, list(X_train.columns)


def save_regression_model(target_name, X, y, feature_cols):
    from xgboost import XGBRegressor
    from utils.metrics import mae, rmse, r2_score

    params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
    }

    model = XGBRegressor(**params)
    t0 = time.time()
    model.fit(X, y)
    train_time = time.time() - t0

    y_pred = model.predict(X)
    metrics = {
        'mae': round(mae(y.values, y_pred), 4),
        'rmse': round(rmse(y.values, y_pred), 4),
        'r2': round(r2_score(y.values, y_pred), 4),
        'train_time': round(train_time, 2),
    }

    model_path = os.path.join(MODEL_DIR, f'{target_name}.pkl')
    meta = {
        'target_name': target_name,
        'target_label': TARGET_LABELS.get(target_name, target_name),
        'target_type': 'regression',
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'n_samples': len(X),
        'metrics': metrics,
    }

    joblib.dump({'model': model, 'meta': meta}, model_path)
    logger.info(f"已保存 {target_name}: R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, "
                f"size={os.path.getsize(model_path) / 1024:.0f}KB")
    return metrics


def save_wind_direction_model(target_name, X, y, feature_cols):
    from xgboost import XGBRegressor
    from utils.metrics import angular_error_from_components

    y_vals = np.array(y)
    y_sin = np.sin(np.radians(y_vals))
    y_cos = np.cos(np.radians(y_vals))

    params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
    }

    model_sin = XGBRegressor(**params)
    model_cos = XGBRegressor(**params)

    t0 = time.time()
    model_sin.fit(X, y_sin)
    model_cos.fit(X, y_cos)
    train_time = time.time() - t0

    sin_pred = model_sin.predict(X)
    cos_pred = model_cos.predict(X)
    angular_err = angular_error_from_components(y_sin, y_cos, sin_pred, cos_pred)

    metrics = {
        'angular_error': round(angular_err, 4),
        'train_time': round(train_time, 2),
    }

    model_path = os.path.join(MODEL_DIR, f'{target_name}.pkl')
    meta = {
        'target_name': target_name,
        'target_label': TARGET_LABELS.get(target_name, target_name),
        'target_type': 'angular_regression',
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'n_samples': len(X),
        'metrics': metrics,
    }

    joblib.dump({'model_sin': model_sin, 'model_cos': model_cos, 'meta': meta}, model_path)
    logger.info(f"已保存 {target_name}: angular_err={metrics['angular_error']:.4f}°, "
                f"size={os.path.getsize(model_path) / 1024:.0f}KB")
    return metrics


def save_rainfall_model(target_name, X, y, feature_cols):
    from xgboost import XGBClassifier, XGBRegressor
    from utils.metrics import auc, brier_score, csi, mae

    y_vals = np.array(y)
    y_flag = (y_vals >= 0.1).astype(int)

    params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
    }

    cls_params = {**params, 'eval_metric': 'logloss'}
    reg_params = {**params}

    cls_model = XGBClassifier(**cls_params)
    reg_model = XGBRegressor(**reg_params)

    t0 = time.time()
    cls_model.fit(X, y_flag)
    rain_mask = y_flag == 1
    if rain_mask.sum() > 0:
        reg_model.fit(X[rain_mask], y_vals[rain_mask])
    train_time = time.time() - t0

    y_prob = cls_model.predict_proba(X)[:, 1]
    y_pred_binary = cls_model.predict(X)

    metrics = {
        'cls_auc': round(auc(y_flag, y_prob), 4),
        'brier': round(brier_score(y_flag, y_prob), 4),
        'csi': round(csi(y_flag, y_pred_binary), 4),
        'train_time': round(train_time, 2),
    }

    model_path = os.path.join(MODEL_DIR, f'{target_name}.pkl')
    meta = {
        'target_name': target_name,
        'target_label': TARGET_LABELS.get(target_name, target_name),
        'target_type': 'two_stage',
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'n_samples': len(X),
        'metrics': metrics,
    }

    joblib.dump({
        'cls_model': cls_model,
        'reg_model': reg_model if rain_mask.sum() > 0 else None,
        'meta': meta,
    }, model_path)
    logger.info(f"已保存 {target_name}: auc={metrics['cls_auc']:.4f}, csi={metrics['csi']:.4f}, "
                f"size={os.path.getsize(model_path) / 1024:.0f}KB")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='训练并保存 XGBoost 模型')
    parser.add_argument('--target', type=str, help='只训练指定目标')
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    targets_to_train = [args.target] if args.target else sorted(TARGET_TYPES.keys())
    available_targets = [d for d in os.listdir(DATA_DIR)
                         if os.path.isdir(os.path.join(DATA_DIR, d)) and d in TARGET_TYPES]

    targets_to_train = [t for t in targets_to_train if t in available_targets]
    if not targets_to_train:
        logger.error("没有可训练的目标")
        return

    logger.info(f"将训练 {len(targets_to_train)} 个目标: {targets_to_train}")
    print(f"\n{'='*60}")
    print(f"  保存 XGBoost 模型到 {MODEL_DIR}/")
    print(f"{'='*60}\n")

    summary = []
    for target_name in targets_to_train:
        try:
            X, y, feature_cols = load_data(target_name)
            target_type = TARGET_TYPES[target_name]

            if target_type == 'regression':
                metrics = save_regression_model(target_name, X, y, feature_cols)
                print(f"  {target_name:30s}  R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
                summary.append((target_name, f"R2={metrics['r2']:.4f}", metrics['train_time']))
            elif target_type == 'angular_regression':
                metrics = save_wind_direction_model(target_name, X, y, feature_cols)
                print(f"  {target_name:30s}  angular_err={metrics['angular_error']:.4f}°")
                summary.append((target_name, f"ang={metrics['angular_error']:.2f}°", metrics['train_time']))
            elif target_type == 'two_stage':
                metrics = save_rainfall_model(target_name, X, y, feature_cols)
                print(f"  {target_name:30s}  auc={metrics['cls_auc']:.4f}, csi={metrics['csi']:.4f}")
                summary.append((target_name, f"auc={metrics['cls_auc']:.4f}", metrics['train_time']))

        except Exception as e:
            logger.error(f"训练失败 {target_name}: {e}")
            import traceback
            traceback.print_exc()

    manifest = {
        'models': [t[0] for t in summary],
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(MODEL_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  完成: {len(summary)}/{len(targets_to_train)} 模型已保存")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()