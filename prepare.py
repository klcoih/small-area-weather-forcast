#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoResearch prepare.py —— 8个预测目标的环境数据准备

预测目标（15分钟间隔）：
  1. greenhouse_temperature   大棚温度     (连续回归)
  2. greenhouse_humidity       大棚湿度     (连续回归)
  3. outdoor_temperature        户外温度     (连续回归)
  4. outdoor_humidity           户外湿度     (连续回归)
  5. light_intensity            光照强度     (连续回归，夜间可能为0)
  6. wind_direction             风向         (向量回归 sin/cos)
  7. wind_speed                 风速         (连续回归)
  8. rainfall                   降雨量       (两阶段: 分类 + 回归)

输出格式:
  train.py 会将每轮训练结果追加到 results.csv:
    model_name,target_name,val_mae,val_rmse,val_r2,val_special,train_time

数据源:
  支持从 preprocessed_data/ CSV 文件加载，或从数据加载模块直接生成
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 0. 全局配置
# ═══════════════════════════════════════════════════════════════

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'preprocessed_data')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'autoresearch_data')

TARGET_CONFIG = {
    'greenhouse_temperature': {
        'internal_name': 'temperature',
        'type': 'regression',
        'unit': '°C',
        'label': '大棚温度',
    },
    'greenhouse_humidity': {
        'internal_name': 'humidity',
        'type': 'regression',
        'unit': '%',
        'label': '大棚湿度',
    },
    'outdoor_temperature': {
        'internal_name': 'temperature_outdoor',
        'type': 'regression',
        'unit': '°C',
        'label': '户外温度',
    },
    'outdoor_humidity': {
        'internal_name': 'humidity_outdoor',
        'type': 'regression',
        'unit': '%',
        'label': '户外湿度',
    },
    'light_intensity': {
        'internal_name': 'light_intensity',
        'type': 'regression',
        'unit': 'lux',
        'label': '光照强度',
    },
    'wind_direction': {
        'internal_name': 'wind_direction',
        'type': 'angular_regression',
        'sub_targets': ['wind_sin', 'wind_cos'],
        'unit': '°',
        'label': '风向',
    },
    'wind_speed': {
        'internal_name': 'wind_speed',
        'type': 'regression',
        'unit': 'm/s',
        'label': '风速',
    },
    'rainfall': {
        'internal_name': 'rainfall',
        'type': 'two_stage',
        'classification_target': 'rain_flag',
        'regression_target': 'rainfall',
        'unit': 'mm',
        'label': '降雨量',
    },
}

SPLIT_CONFIG = {
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'method': 'time_series',
}

BASELINE_METRICS = {
    'greenhouse_temperature': {
        'naive_last':   {'mae': 1.85, 'rmse': 2.52, 'r2': 0.81},
        'naive_seasonal': {'mae': 2.10, 'rmse': 2.85, 'r2': 0.76},
        'linear_regression': {'mae': 1.10, 'rmse': 1.50, 'r2': 0.91},
    },
    'greenhouse_humidity': {
        'naive_last':   {'mae': 2.80, 'rmse': 3.60, 'r2': 0.78},
        'naive_seasonal': {'mae': 3.20, 'rmse': 4.10, 'r2': 0.72},
        'linear_regression': {'mae': 1.80, 'rmse': 2.40, 'r2': 0.88},
    },
    'outdoor_temperature': {
        'naive_last':   {'mae': 0.65, 'rmse': 0.95, 'r2': 0.94},
        'naive_seasonal': {'mae': 0.85, 'rmse': 1.20, 'r2': 0.90},
        'linear_regression': {'mae': 0.40, 'rmse': 0.60, 'r2': 0.97},
    },
    'outdoor_humidity': {
        'naive_last':   {'mae': 3.20, 'rmse': 4.50, 'r2': 0.75},
        'naive_seasonal': {'mae': 4.00, 'rmse': 5.30, 'r2': 0.68},
        'linear_regression': {'mae': 2.10, 'rmse': 2.90, 'r2': 0.85},
    },
    'light_intensity': {
        'naive_last':   {'mae': 3200, 'rmse': 5800, 'r2': 0.72},
        'naive_seasonal': {'mae': 2800, 'rmse': 5200, 'r2': 0.78},
        'linear_regression': {'mae': 1800, 'rmse': 3500, 'r2': 0.88},
    },
    'wind_direction': {
        'naive_last':   {'angular_error': 32.0},
        'naive_seasonal': {'angular_error': 28.0},
        'linear_regression': {'angular_error': 15.0},
    },
    'wind_speed': {
        'naive_last':   {'mae': 0.45, 'rmse': 0.62, 'r2': 0.58},
        'naive_seasonal': {'mae': 0.50, 'rmse': 0.68, 'r2': 0.52},
        'linear_regression': {'mae': 0.30, 'rmse': 0.42, 'r2': 0.76},
    },
    'rainfall': {
        'classification': {
            'naive_zero': {'auc': 0.50, 'brier': 0.04, 'csi': 0.0},
            'logistic':   {'auc': 0.85, 'brier': 0.02, 'csi': 0.45},
        },
        'regression': {
            'naive_mean': {'mae_rain_only': 1.50},
            'linear':     {'mae_rain_only': 0.80},
        },
    },
}

# ═══════════════════════════════════════════════════════════════
# 1. 评估指标
# ═══════════════════════════════════════════════════════════════

class Metrics:
    """评估指标实现"""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)

    @classmethod
    def regression_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """通用回归指标"""
        return {
            'mae': round(cls.mae(y_true, y_pred), 6),
            'rmse': round(cls.rmse(y_true, y_pred), 6),
            'r2': round(cls.r2(y_true, y_pred), 6),
        }

    @staticmethod
    def angular_error(angle_true: np.ndarray, angle_pred: np.ndarray) -> float:
        """
        平均绝对角度误差（考虑环形特性）

        Args:
            angle_true: 真实角度 (0-360)
            angle_pred: 预测角度 (0-360)

        Returns:
            float: 平均绝对角度误差（度）
        """
        diff = np.abs(angle_true - angle_pred) % 360
        diff = np.minimum(diff, 360 - diff)
        return float(np.mean(diff))

    @staticmethod
    def angular_error_from_components(sin_true: np.ndarray, cos_true: np.ndarray,
                                      sin_pred: np.ndarray, cos_pred: np.ndarray) -> float:
        """
        从 sin/cos 分量计算角度误差

        Args:
            sin_true, cos_true: 真实值
            sin_pred, cos_pred: 预测值

        Returns:
            float: 平均绝对角度误差（度）
        """
        angle_true = np.degrees(np.arctan2(sin_true, cos_true)) % 360
        angle_pred = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360
        return Metrics.angular_error(angle_true, angle_pred)

    @staticmethod
    def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        ROC AUC（简化实现，不依赖 sklearn 时使用）
        优先使用 sklearn 实现
        """
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) < 2:
                return 0.5
            return float(roc_auc_score(y_true, y_score))
        except ImportError:
            if len(np.unique(y_true)) < 2:
                return 0.5
            desc = np.argsort(y_score)[::-1]
            y_true_sorted = y_true[desc]
            n_pos = np.sum(y_true_sorted == 1)
            n_neg = np.sum(y_true_sorted == 0)
            if n_pos == 0 or n_neg == 0:
                return 0.5
            tpr = np.cumsum(y_true_sorted == 1) / n_pos
            fpr = np.cumsum(y_true_sorted == 0) / n_neg
            return float(np.trapz(tpr, fpr))

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Brier 分数"""
        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def csi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Critical Success Index (CSI) = TP / (TP + FP + FN)
        适用于稀疏事件（降雨检测）
        """
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        denom = tp + fp + fn
        return float(tp / denom) if denom > 0 else 0.0

    @staticmethod
    def composite_score(metrics: Dict, target_type: str) -> float:
        """
        计算综合得分 val_composite（用于 AutoResearch 主判断）

        - 回归目标: 0.3×(-MAE_norm) + 0.3×(-RMSE_norm) + 0.4×R²
        - 角度目标: 1.0×(1 - angular_error/180)
        - 两阶段目标: 0.5×classification_score + 0.5×regression_score
        """
        if target_type == 'regression':
            mae_norm = min(1.0, metrics.get('mae', 10) / 20)
            rmse_norm = min(1.0, metrics.get('rmse', 10) / 25)
            r2 = max(0.0, metrics.get('r2', 0))
            return round(0.3 * (1 - mae_norm) + 0.3 * (1 - rmse_norm) + 0.4 * r2, 4)

        elif target_type == 'angular_regression':
            angular_error = metrics.get('angular_error', 90)
            return round(max(0.0, 1.0 - angular_error / 180.0), 4)

        elif target_type == 'two_stage':
            cls = metrics.get('classification', {})
            reg = metrics.get('regression', {})
            cls_score = (
                0.4 * cls.get('auc', 0) +
                0.3 * (1 - min(1.0, cls.get('brier', 1))) +
                0.3 * cls.get('csi', 0)
            )
            mae_rain = reg.get('mae_rain_only', 5)
            reg_score = max(0.0, 1.0 - mae_rain / 10.0)
            return round(0.5 * cls_score + 0.5 * reg_score, 4)

        return 0.0


# ═══════════════════════════════════════════════════════════════
# 2. 数据加载
# ═══════════════════════════════════════════════════════════════

def load_from_csv(data_dir: str = DATA_DIR) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    从 preprocessed_data/ CSV 文件加载数据

    Returns:
        {
            'greenhouse_temperature': {
                'X_train': DataFrame, 'y_train': Series,
                'X_val': DataFrame,   'y_val': Series,
                'X_test': DataFrame,  'y_test': Series,
            },
            ...
        }
    """
    datasets = {}
    not_found = []

    for ar_name, cfg in TARGET_CONFIG.items():
        internal_name = cfg['internal_name']
        target_dir = os.path.join(data_dir, internal_name)

        if not os.path.isdir(target_dir):
            not_found.append(internal_name)
            continue

        try:
            X_train = pd.read_csv(os.path.join(target_dir, 'X_train.csv'))
            y_train = pd.read_csv(os.path.join(target_dir, 'y_train.csv'))['y']
            X_val = pd.read_csv(os.path.join(target_dir, 'X_val.csv'))
            y_val = pd.read_csv(os.path.join(target_dir, 'y_val.csv'))['y']
            X_test = pd.read_csv(os.path.join(target_dir, 'X_test.csv'))
            y_test = pd.read_csv(os.path.join(target_dir, 'y_test.csv'))['y']

            datasets[ar_name] = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val,     'y_val': y_val,
                'X_test': X_test,   'y_test': y_test,
            }
            logger.info(
                f"加载 {ar_name}: train={X_train.shape}, "
                f"val={X_val.shape}, test={X_test.shape}"
            )
        except Exception as e:
            logger.error(f"加载 {ar_name} 失败: {e}")
            not_found.append(internal_name)

    if not_found:
        logger.warning(f"以下目标数据未找到: {not_found}")

    return datasets


def load_from_pipeline() -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    从 PreprocessingPipeline 直接生成数据（需要先运行 data_preprocessing.py）

    Returns:
        同 load_from_csv
    """
    from data_preprocessing import PreprocessingPipeline
    from greenhouse_data_loader import GreenhouseDataLoader
    from outdoor_weather_loader import OutdoorWeatherLoader

    pipeline = PreprocessingPipeline(output_dir=DATA_DIR)

    gh_path = os.path.join(CURRENT_DIR, '天气数据', '温湿度数据', '温湿度数据.csv')
    ow_path = os.path.join(CURRENT_DIR, '天气数据', '宣威市尚营种气象', '宣威市尚营种气象.csv')

    gh_loader = GreenhouseDataLoader(gh_path)
    gh_loader.load(); gh_loader.check_quality(); gh_loader.standardize()

    ow_loader = OutdoorWeatherLoader(ow_path)
    ow_loader.load(); ow_loader.check_quality(); ow_loader.standardize()

    pipeline.run(gh_loader.df_standardized, ow_loader.df_standardized)

    targets = list(TARGET_CONFIG.keys())
    pipeline.export(targets=targets)

    return load_from_csv(DATA_DIR)


def get_data_summary(datasets: Dict) -> pd.DataFrame:
    """生成数据摘要表"""
    rows = []
    for ar_name, cfg in TARGET_CONFIG.items():
        if ar_name not in datasets:
            continue
        data = datasets[ar_name]
        y_all = pd.concat([data['y_train'], data['y_val'], data['y_test']])
        rows.append({
            'target': ar_name,
            'type': cfg['type'],
            'unit': cfg['unit'],
            'train_samples': len(data['y_train']),
            'val_samples': len(data['y_val']),
            'test_samples': len(data['y_test']),
            'features': data['X_train'].shape[1],
            'target_mean': round(y_all.mean(), 3),
            'target_std': round(y_all.std(), 3),
            'target_min': round(y_all.min(), 3),
            'target_max': round(y_all.max(), 3),
        })
    return pd.DataFrame(rows)


def save_prepared_data(datasets: Dict, output_dir: str = OUTPUT_DIR):
    """
    保存准备好的数据供 train.py 读取

    输出结构:
      autoresearch_data/
        └── greenhouse_temperature/
            ├── X_train.csv
            ├── y_train.csv
            ├── X_val.csv
            ├── y_val.csv
            ├── X_test.csv
            ├── y_test.csv
            └── config.json
    """
    os.makedirs(output_dir, exist_ok=True)

    for ar_name, data in datasets.items():
        target_dir = os.path.join(output_dir, ar_name)
        os.makedirs(target_dir, exist_ok=True)

        data['X_train'].to_csv(os.path.join(target_dir, 'X_train.csv'), index=False)
        data['y_train'].to_csv(os.path.join(target_dir, 'y_train.csv'), index=False, header=['y'])
        data['X_val'].to_csv(os.path.join(target_dir, 'X_val.csv'), index=False)
        data['y_val'].to_csv(os.path.join(target_dir, 'y_val.csv'), index=False, header=['y'])
        data['X_test'].to_csv(os.path.join(target_dir, 'X_test.csv'), index=False)
        data['y_test'].to_csv(os.path.join(target_dir, 'y_test.csv'), index=False, header=['y'])

        cfg = TARGET_CONFIG[ar_name].copy()
        cfg['features'] = list(data['X_train'].columns)
        cfg['n_features'] = data['X_train'].shape[1]
        cfg['n_train'] = len(data['y_train'])
        cfg['n_val'] = len(data['y_val'])
        cfg['n_test'] = len(data['y_test'])
        with open(os.path.join(target_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        logger.info(f"已保存 {ar_name}: {target_dir}")

    summary = {
        'targets': list(datasets.keys()),
        'split_config': SPLIT_CONFIG,
        'baseline_metrics': BASELINE_METRICS,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"全部数据已保存到 {output_dir}")


# ═══════════════════════════════════════════════════════════════
# 3. train.py 输出格式 & 评估器
# ═══════════════════════════════════════════════════════════════

class ResultEvaluator:
    """
    评估器：计算各目标的指标并输出结果

    train.py 应在训练完成后调用本评估器的 evaluate() 方法，
    将结果追加写入 results.csv：

    model_name,target_name,val_mae,val_rmse,val_r2,val_special,train_time
    """

    def __init__(self, results_csv: str = 'results.csv'):
        self.results_csv = results_csv
        self._init_results_file()

    def _init_results_file(self):
        if not os.path.exists(self.results_csv):
            with open(self.results_csv, 'w', encoding='utf-8') as f:
                f.write('model_name,target_name,val_mae,val_rmse,val_r2,val_special,train_time\n')

    def evaluate(self, model_name: str, target_name: str,
                 y_true: np.ndarray, y_pred: np.ndarray,
                 train_time: float,
                 y_true_extra: Optional[np.ndarray] = None,
                 y_pred_extra: Optional[np.ndarray] = None) -> Dict:
        """
        评估模型并写入结果

        Args:
            model_name:   模型名称 (XGBoost, LSTM, Prophet, ...)
            target_name:  目标名称 (greenhouse_temperature, ...)
            y_true:       真实值
            y_pred:       预测值
            train_time:   训练耗时（秒）
            y_true_extra: 额外真实值（如 rainfall 回归阶段的真实降雨量）
            y_pred_extra: 额外预测值

        Returns:
            dict: 完整评估指标
        """
        cfg = TARGET_CONFIG.get(target_name, {})
        target_type = cfg.get('type', 'regression')

        if target_type == 'regression':
            return self._eval_regression(model_name, target_name,
                                         y_true, y_pred, train_time)

        elif target_type == 'angular_regression':
            return self._eval_angular(model_name, target_name,
                                      y_true, y_pred, train_time)

        elif target_type == 'two_stage':
            return self._eval_two_stage(model_name, target_name,
                                        y_true, y_pred, train_time,
                                        y_true_extra, y_pred_extra)

        return {}

    def _eval_regression(self, model_name: str, target_name: str,
                         y_true: np.ndarray, y_pred: np.ndarray,
                         train_time: float) -> Dict:
        m = Metrics.regression_metrics(y_true, y_pred)
        composite = Metrics.composite_score(m, 'regression')

        special = '-'
        self._write_result(model_name, target_name,
                           m['mae'], m['rmse'], m['r2'], special, train_time)

        m['val_composite'] = composite
        return m

    def _eval_angular(self, model_name: str, target_name: str,
                      y_true: np.ndarray, y_pred: np.ndarray,
                      train_time: float) -> Dict:
        """
        风向评估：
          y_true: (N, 2) [sin, cos] 或 (N,) 角度
          y_pred: (N, 2) [sin, cos] 或 (N,) 角度
        """
        if y_true.ndim == 2 and y_true.shape[1] == 2:
            sin_t, cos_t = y_true[:, 0], y_true[:, 1]
            sin_p, cos_p = y_pred[:, 0], y_pred[:, 1]
            angular_err = Metrics.angular_error_from_components(sin_t, cos_t, sin_p, cos_p)
        else:
            angular_err = Metrics.angular_error(y_true.flatten(), y_pred.flatten())

        mae_sin = Metrics.mae(
            y_true[:, 0] if y_true.ndim == 2 else np.sin(np.radians(y_true)),
            y_pred[:, 0] if y_pred.ndim == 2 else np.sin(np.radians(y_pred))
        )
        mae_cos = Metrics.mae(
            y_true[:, 1] if y_true.ndim == 2 else np.cos(np.radians(y_true)),
            y_pred[:, 1] if y_pred.ndim == 2 else np.cos(np.radians(y_pred))
        )

        special = f"angular_err={angular_err:.3f}"
        self._write_result(model_name, target_name,
                           round((mae_sin + mae_cos) / 2, 4), '-', '-', special, train_time)

        composite = Metrics.composite_score({'angular_error': angular_err}, 'angular_regression')
        return {
            'angular_error': round(angular_err, 3),
            'mae_sin': round(mae_sin, 4),
            'mae_cos': round(mae_cos, 4),
            'val_composite': composite,
        }

    def _eval_two_stage(self, model_name: str, target_name: str,
                        y_true: np.ndarray, y_pred_class: np.ndarray,
                        train_time: float,
                        y_true_rain: np.ndarray = None,
                        y_pred_rain: np.ndarray = None) -> Dict:
        """
        降雨两阶段评估：
          - 分类阶段: y_true (0/1), y_pred (概率 or 0/1)
          - 回归阶段: y_true_rain (降雨量), y_pred_rain (预测降雨量)
        """
        if y_pred_class.ndim > 1 and y_pred_class.shape[1] == 2:
            y_prob = y_pred_class[:, 1]
            y_pred_binary = (y_prob >= 0.5).astype(int)
        else:
            y_prob = y_pred_class.flatten()
            y_pred_binary = (y_prob >= 0.5).astype(int)

        y_true_binary = y_true.flatten().astype(int)

        cls_auc = Metrics.auc(y_true_binary, y_prob)
        cls_brier = Metrics.brier_score(y_true_binary, y_prob)
        cls_csi = Metrics.csi(y_true_binary, y_pred_binary)

        rain_mae = '-'
        if y_true_rain is not None and y_pred_rain is not None:
            rain_mask = y_true_binary == 1
            if rain_mask.sum() > 0:
                rain_mae = round(
                    Metrics.mae(y_true_rain[rain_mask], y_pred_rain[rain_mask]), 4
                )

        special = (
            f"cls_auc={cls_auc:.4f}|brier={cls_brier:.4f}|"
            f"csi={cls_csi:.4f}|rain_mae={rain_mae}"
        )
        self._write_result(model_name, target_name, '-', '-', '-', special, train_time)

        cls_metrics = {'auc': cls_auc, 'brier': cls_brier, 'csi': cls_csi}
        reg_metrics = {'mae_rain_only': rain_mae if rain_mae != '-' else 0}
        composite = Metrics.composite_score(
            {'classification': cls_metrics, 'regression': reg_metrics}, 'two_stage'
        )

        return {
            'classification': cls_metrics,
            'regression': reg_metrics,
            'val_composite': composite,
        }

    def _write_result(self, model_name: str, target_name: str,
                      val_mae, val_rmse, val_r2, val_special, train_time):
        with open(self.results_csv, 'a', encoding='utf-8') as f:
            f.write(f"{model_name},{target_name},{val_mae},{val_rmse},{val_r2},{val_special},{train_time}\n")

    def read_results(self) -> pd.DataFrame:
        """读取已有结果"""
        if not os.path.exists(self.results_csv):
            return pd.DataFrame()
        return pd.read_csv(self.results_csv)

    def get_best_per_target(self) -> pd.DataFrame:
        """获取每目标的最佳结果（按 val_composite 排序）"""
        df = self.read_results()
        if df.empty:
            return df
        try:
            df['val_composite'] = df.apply(
                lambda row: self._compute_composite_from_row(row), axis=1
            )
            return df.sort_values('val_composite', ascending=False)
        except Exception:
            return df.sort_values('val_r2', ascending=False)

    def _compute_composite_from_row(self, row) -> float:
        target = row.get('target_name', '')
        cfg = TARGET_CONFIG.get(target, {})
        target_type = cfg.get('type', 'regression')

        if target_type == 'regression':
            metrics = {'mae': row.get('val_mae', 0) if row.get('val_mae') != '-' else 0,
                       'rmse': row.get('val_rmse', 0) if row.get('val_rmse') != '-' else 0,
                       'r2': row.get('val_r2', 0) if row.get('val_r2') != '-' else 0}
            return Metrics.composite_score(metrics, 'regression')

        return 0.0


# ═══════════════════════════════════════════════════════════════
# 4. 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    """
    prepare.py 主入口 —— AutoResearch 框架调用此函数准备数据

    流程:
      1. 加载数据 (CSV / Pipeline)
      2. 验证数据完整性
      3. 保存到 autoresearch_data/
      4. 打印摘要
    """
    print("=" * 60)
    print("  AutoResearch prepare.py —— 数据准备")
    print("=" * 60)

    print(f"\n数据源目录: {DATA_DIR}")
    print(f"输出目录:   {OUTPUT_DIR}")

    # 1. 加载数据
    print("\n[1] 加载预处理数据...")
    datasets = load_from_csv(DATA_DIR)

    if not datasets:
        print("\n[!] 未找到预处理数据，尝试从 Pipeline 生成...")
        try:
            datasets = load_from_pipeline()
        except Exception as e:
            logger.error(f"Pipeline 生成失败: {e}")
            print("请先运行 data_preprocessing.py 生成预处理数据")
            return 1

    print(f"\n已加载 {len(datasets)}/{len(TARGET_CONFIG)} 个目标")

    # 2. 数据摘要
    print("\n[2] 数据摘要")
    print("-" * 60)
    summary_df = get_data_summary(datasets)
    print(summary_df.to_string(index=False))

    # 3. 基线指标
    print("\n[3] 基线指标参考")
    print("-" * 60)
    for target_name, baselines in BASELINE_METRICS.items():
        if target_name not in datasets:
            continue
        cfg = TARGET_CONFIG[target_name]
        print(f"\n  [{target_name}] ({cfg['label']})")
        t = cfg['type']
        if t == 'regression':
            for model, m in baselines.items():
                print("    {:<20s}: MAE={:>8}, RMSE={:>8}, R2={:>6}".format(
                    model, str(m.get('mae', '-')), str(m.get('rmse', '-')), str(m.get('r2', '-'))))
        elif t == 'angular_regression':
            for model, m in baselines.items():
                print("    {:<20s}: angular_error={:>6}deg".format(
                    model, str(m.get('angular_error', '-'))))
        elif t == 'two_stage':
            print("    classification baseline:")
            for model, m in baselines.get('classification', {}).items():
                print("      {:<18s}: AUC={:>6}, Brier={:>6}, CSI={:>6}".format(
                    model, str(m.get('auc', '-')), str(m.get('brier', '-')), str(m.get('csi', '-'))))
            print("    regression baseline:")
            for model, m in baselines.get('regression', {}).items():
                print("      {:<18s}: MAE(rain)={:>8}".format(
                    model, str(m.get('mae_rain_only', '-'))))

    # 4. 保存
    print("\n[4] 保存准备好的数据...")
    save_prepared_data(datasets, OUTPUT_DIR)

    # 5. 下一步提示
    print("\n" + "=" * 60)
    print("  数据准备完成！")
    print("=" * 60)
    print(f"\n  数据已保存到: {OUTPUT_DIR}/")
    print(f"  配置摘要:     {OUTPUT_DIR}/summary.json")
    print(f"\n  下一步: 编写 train.py 进行模型训练")
    print(f"  训练结果将写入: {os.path.join(CURRENT_DIR, 'results.csv')}")
    print(f"\n  预期 train.py 输出格式:")
    print(f"    model_name,target_name,val_mae,val_rmse,val_r2,val_special,train_time")
    print(f"    XGBoost,greenhouse_temperature,0.5,0.8,0.92,-,45.2")

    return 0


if __name__ == "__main__":
    sys.exit(main())