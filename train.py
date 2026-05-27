#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoResearch train.py —— 模型训练主程序

流程:
  1. 读取 prepare.py 生成的训练/验证数据
  2. 根据当前配置选择模型类型
  3. 训练模型
  4. 在验证集上评估
  5. 输出评估结果到 results.csv

输出格式:
  model_name,target_name,val_mae,val_rmse,val_r2,val_special,train_time

支持模型:
  - arima:  ARIMA (smdarima auto_arima)
  - prophet: Facebook Prophet
  - xgboost: XGBoost
  - lstm:   PyTorch LSTM
  - markov:  Markov Chain
"""

import os
import sys
import json
import time
import argparse
import logging
import warnings
import traceback
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'autoresearch_data')

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


def load_data(target_name, data_dir=DATA_DIR):
    """加载 prepare.py 导出的训练/验证数据"""
    target_dir = os.path.join(data_dir, target_name)
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"目标数据目录不存在: {target_dir}")

    X_train = pd.read_csv(os.path.join(target_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(target_dir, 'y_train.csv'))['y']
    X_val = pd.read_csv(os.path.join(target_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(target_dir, 'y_val.csv'))['y']

    logger.info(f"加载 {target_name}: train={X_train.shape}, val={X_val.shape}")
    return X_train, y_train, X_val, y_val


def train_regression(model_name, target_name, X_train, y_train, X_val, y_val, params=None):
    """训练回归模型"""
    if params is None:
        params = {}

    if model_name == 'arima':
        from models.arima_model import train_arima
        y_pred, metrics, t = train_arima(y_train.values, y_val.values, params, return_train_time=True)
        return y_pred, metrics, t

    elif model_name == 'prophet':
        from models.prophet_model import train_prophet
        y_pred, metrics, t = train_prophet(y_train, y_val, params=params)
        return y_pred, metrics, t

    elif model_name == 'xgboost':
        from models.xgboost_model import train_xgboost_regression
        return train_xgboost_regression(X_train, y_train, X_val, y_val, params)

    elif model_name == 'lstm':
        from models.lstm_model import train_lstm_regression
        return train_lstm_regression(X_train, y_train, X_val, y_val, params)

    else:
        raise ValueError(f"未知模型: {model_name}")


def train_angular(model_name, target_name, X_train, y_train, X_val, y_val, params=None):
    """训练风向模型 (sin/cos)"""
    if params is None:
        params = {}

    y_train_vals = np.array(y_train)
    y_val_vals = np.array(y_val)

    y_train_sin = np.sin(np.radians(y_train_vals))
    y_train_cos = np.cos(np.radians(y_train_vals))
    y_val_sin = np.sin(np.radians(y_val_vals))
    y_val_cos = np.cos(np.radians(y_val_vals))

    if model_name == 'xgboost':
        from models.xgboost_model import train_xgboost_wind
        return train_xgboost_wind(X_train, y_train_sin, y_train_cos,
                                  X_val, y_val_sin, y_val_cos, params)

    elif model_name == 'lstm':
        from models.lstm_model import train_lstm_wind
        return train_lstm_wind(X_train, y_train_sin, y_train_cos,
                               X_val, y_val_sin, y_val_cos, params)

    else:
        raise ValueError(f"风向不支持模型: {model_name}")


def train_two_stage(model_name, target_name, X_train, y_train, X_val, y_val, params=None):
    """训练降雨两阶段模型"""
    if params is None:
        params = {}

    y_train_vals = np.array(y_train)
    y_val_vals = np.array(y_val)
    y_train_flag = (y_train_vals >= 0.1).astype(int)
    y_val_flag = (y_val_vals >= 0.1).astype(int)

    if model_name == 'xgboost':
        from models.xgboost_model import train_xgboost_rainfall_two_stage
        return train_xgboost_rainfall_two_stage(
            X_train, y_train_flag, y_train_vals,
            X_val, y_val_flag, y_val_vals, params
        )

    elif model_name == 'markov':
        from models.markov_model import train_markov
        pred_states, metrics, t = train_markov(y_train_vals, y_val_vals, params)
        return pred_states, metrics, t

    elif model_name == 'lstm':
        from models.lstm_model import train_lstm_rainfall_two_stage
        return train_lstm_rainfall_two_stage(
            X_train, y_train_flag, y_train_vals,
            X_val, y_val_flag, y_val_vals, params
        )

    else:
        raise ValueError(f"降雨不支持模型: {model_name}")


def format_special(target_type, metrics):
    """格式化 val_special 字段"""
    if target_type == 'regression':
        return '-'
    elif target_type == 'angular_regression':
        return f"angular_err={metrics.get('angular_error', '-')}"
    elif target_type == 'two_stage':
        cls = metrics.get('classification', {})
        reg = metrics.get('regression', {})
        return (f"cls_auc={cls.get('auc', '-'):.4f}|"
                f"brier={cls.get('brier', '-'):.4f}|"
                f"csi={cls.get('csi', '-'):.4f}|"
                f"rain_mae={reg.get('mae_rain_only', '-')}")
    return '-'


def write_result(results_csv, model_name, target_name, metrics, train_time, target_type):
    """写入结果到 CSV"""
    val_mae = metrics.get('mae', '-')
    val_rmse = metrics.get('rmse', '-')
    val_r2 = metrics.get('r2', '-')
    val_special = format_special(target_type, metrics)

    if not os.path.exists(results_csv):
        with open(results_csv, 'w', encoding='utf-8') as f:
            f.write('model_name,target_name,val_mae,val_rmse,val_r2,val_special,train_time\n')

    line = f"{model_name},{target_name},{val_mae},{val_rmse},{val_r2},{val_special},{train_time:.2f}\n"
    with open(results_csv, 'a', encoding='utf-8') as f:
        f.write(line)

    logger.info(f"结果写入: {model_name} @ {target_name}")


def run_single(model_name, target_name, params=None, results_csv='results.csv', data_dir=DATA_DIR):
    """
    训练单个 (模型, 目标) 组合

    Args:
        model_name:  arima / prophet / xgboost / lstm / markov
        target_name: greenhouse_temperature / ...
        params:      模型参数字典
        results_csv: 输出 CSV 路径
        data_dir:    数据目录
    """
    import random
    random.seed(42)
    np.random.seed(42)

    target_type = TARGET_TYPES.get(target_name, 'regression')

    logger.info(f"=" * 50)
    logger.info(f"训练: {model_name} -> {target_name} (type={target_type})")
    logger.info(f"=" * 50)

    try:
        X_train, y_train, X_val, y_val = load_data(target_name, data_dir)

        if target_type == 'regression':
            y_pred, metrics, train_time = train_regression(
                model_name, target_name, X_train, y_train, X_val, y_val, params
            )
        elif target_type == 'angular_regression':
            y_pred, metrics, train_time = train_angular(
                model_name, target_name, X_train, y_train, X_val, y_val, params
            )
        elif target_type == 'two_stage':
            y_pred, metrics, train_time = train_two_stage(
                model_name, target_name, X_train, y_train, X_val, y_val, params
            )
        else:
            raise ValueError(f"未知目标类型: {target_type}")

        write_result(results_csv, model_name, target_name, metrics, train_time, target_type)

        logger.info(f"训练完成: train_time={train_time:.2f}s")
        logger.info(f"指标: {metrics}")
        return True

    except Exception as e:
        logger.error(f"训练失败 {model_name} @ {target_name}: {e}")
        traceback.print_exc()
        return False


def get_available_targets(data_dir=DATA_DIR):
    """获取 data_dir 下可用的目标列表"""
    targets = []
    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            target_dir = os.path.join(data_dir, name)
            if os.path.isdir(target_dir) and name in TARGET_TYPES:
                if os.path.exists(os.path.join(target_dir, 'X_train.csv')):
                    targets.append(name)
    return sorted(targets)


def get_available_models():
    """检测可用的模型"""
    models = []

    try:
        import pmdarima
        models.append('arima')
    except ImportError:
        pass

    try:
        import prophet
        models.append('prophet')
    except ImportError:
        pass

    try:
        import xgboost
        models.append('xgboost')
    except ImportError:
        pass

    try:
        import torch
        models.append('lstm')
    except ImportError:
        pass

    models.append('markov')
    return models


def main():
    parser = argparse.ArgumentParser(description='AutoResearch train.py - 模型训练')
    parser.add_argument('--model', type=str, help='模型名称 (arima/prophet/xgboost/lstm/markov)')
    parser.add_argument('--target', type=str, help='目标名称')
    parser.add_argument('--params', type=str, help='JSON 参数字符串', default='{}')
    parser.add_argument('--results', type=str, default='results.csv', help='结果输出 CSV')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='数据目录')
    parser.add_argument('--list-models', action='store_true', help='列出可用模型')
    parser.add_argument('--list-targets', action='store_true', help='列出可用目标')
    parser.add_argument('--auto', action='store_true', help='自动运行所有 (模型×目标) 组合')

    args = parser.parse_args()

    if args.list_models:
        print("可用模型:", ", ".join(get_available_models()))
        return

    if args.list_targets:
        print("可用目标:", ", ".join(get_available_targets(args.data_dir)))
        return

    if args.auto:
        models = get_available_models()
        targets = get_available_targets(args.data_dir)
        logger.info(f"自动模式: {len(models)} 模型 × {len(targets)} 目标")

        for model in models:
            for target in targets:
                target_type = TARGET_TYPES.get(target, 'regression')

                if model == 'arima' and target_type not in ('regression',):
                    continue
                if model == 'prophet' and target_type not in ('regression',):
                    continue
                if model == 'markov' and target_type not in ('two_stage',):
                    continue

                run_single(model, target, results_csv=args.results, data_dir=args.data_dir)
        return

    if args.model and args.target:
        params = json.loads(args.params) if args.params else {}
        success = run_single(
            args.model, args.target, params=params,
            results_csv=args.results, data_dir=args.data_dir
        )
        if not success:
            sys.exit(1)
        return

    parser.print_help()


if __name__ == '__main__':
    main()