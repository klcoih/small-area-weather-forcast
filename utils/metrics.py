#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估指标工具"""

import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def r2_score(y_true, y_pred):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def angular_error(angle_true, angle_pred):
    diff = np.abs(np.array(angle_true) - np.array(angle_pred)) % 360
    diff = np.minimum(diff, 360 - diff)
    return float(np.mean(diff))


def angular_error_from_components(sin_true, cos_true, sin_pred, cos_pred):
    angle_true = np.degrees(np.arctan2(np.array(sin_true), np.array(cos_true))) % 360
    angle_pred = np.degrees(np.arctan2(np.array(sin_pred), np.array(cos_pred))) % 360
    return angular_error(angle_true, angle_pred)


def auc(y_true, y_score):
    try:
        from sklearn.metrics import roc_auc_score
        yt = np.array(y_true, dtype=int)
        if len(np.unique(yt)) < 2:
            return 0.5
        return float(roc_auc_score(yt, y_score))
    except ImportError:
        order = np.argsort(np.array(y_score))[::-1]
        yt = np.array(y_true, dtype=int)[order]
        n_pos = np.sum(yt == 1)
        n_neg = np.sum(yt == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tpr = np.cumsum(yt == 1) / n_pos
        fpr = np.cumsum(yt == 0) / n_neg
        return float(np.trapz(tpr, fpr))


def brier_score(y_true, y_prob):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_prob, dtype=float)
    return float(np.mean((yp - yt) ** 2))


def csi(y_true, y_pred):
    yt = np.array(y_true, dtype=int)
    yp = np.array(y_pred, dtype=int)
    tp = np.sum((yp == 1) & (yt == 1))
    fp = np.sum((yp == 1) & (yt == 0))
    fn = np.sum((yp == 0) & (yt == 1))
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0


def composite_score(target_type, metrics_dict):
    if target_type == 'regression':
        mae_norm = min(1.0, metrics_dict.get('mae', 10) / 20)
        rmse_norm = min(1.0, metrics_dict.get('rmse', 10) / 25)
        r2 = max(0.0, metrics_dict.get('r2', 0))
        return round(0.3 * (1 - mae_norm) + 0.3 * (1 - rmse_norm) + 0.4 * r2, 4)
    elif target_type == 'angular_regression':
        ae = metrics_dict.get('angular_error', 90)
        return round(max(0.0, 1.0 - ae / 180.0), 4)
    elif target_type == 'two_stage':
        cls = metrics_dict.get('classification', {})
        reg = metrics_dict.get('regression', {})
        cls_score = (0.4 * cls.get('auc', 0) +
                     0.3 * (1 - min(1.0, cls.get('brier', 1))) +
                     0.3 * cls.get('csi', 0))
        mae_rain = reg.get('mae_rain_only', 5)
        reg_score = max(0.0, 1.0 - mae_rain / 10.0)
        return round(0.5 * cls_score + 0.5 * reg_score, 4)
    return 0.0