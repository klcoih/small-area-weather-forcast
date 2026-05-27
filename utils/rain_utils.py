#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
降雨工具函数 — 向后兼容入口
所有功能已迁移至 utils.rainfall_utils
"""

from utils.rainfall_utils import (
    RAIN_LABELS_4,
    rain_binary_label,
    rain_categories,
    rain_zero_inflation_ratio,
    rain_sample_weight,
    split_rain_data,
    log1p_transform,
    expm1_inverse,
    compute_scale_pos_weight,
    generate_rainfall_features,
    train_rainfall_model,
    train_rainfall_markov,
    predict_rainfall,
    evaluate_rainfall,
    train_rainfall,
)