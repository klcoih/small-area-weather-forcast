#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理配置文件 —— 15分钟间隔环境数据
"""

PREPROCESS_CONFIG = {

    "time": {
        "frequency": "15min",
        "freq_minutes": 15,
        "fill_method": "linear",
        "duplicate_method": "mean",
    },

    "cleaning": {
        "anomaly_method": "iqr",
        "iqr_factor": 1.5,
        "sigma_factor": 3.0,
        "missing_method": "linear",
        "smoothing": False,
        "smoothing_window": 3,
        "smoothing_method": "rolling_mean",
    },

    "features": {

        "lag": {
            "short": [1, 2, 4],
            "medium": [96, 192],
            "long": [672],
        },

        "time_features": [
            "hour",
            "minute",
            "day_of_week",
            "month",
            "is_daytime",
            "hour_sin",
            "hour_cos",
        ],

        "rolling": {
            "mean": [4, 16],
            "std": [4],
            "min": [],
            "max": [],
        },
    },

    "special_preprocessors": {

        "wind_direction": {
            "enabled": True,
            "transform": "sin_cos",
        },

        "rainfall": {
            "enabled": True,
            "binary_threshold": 0.1,
            "binary_label": "rain_flag",
        },

        "light_intensity": {
            "enabled": True,
            "night_threshold": 100.0,
            "night_label": "is_night",
        },
    },

    "cross_features": {
        "temp_diff": True,
        "humidity_diff": True,
        "temp_humidity_interaction": True,
    },

    "export": {

        "targets": [
            "temperature",
            "humidity",
            "temperature_outdoor",
            "humidity_outdoor",
            "light_intensity",
            "wind_direction",
            "wind_speed",
            "rainfall",
        ],

        "split": {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "method": "time_series",
        },

        "output_format": "csv",
        "output_dir": "preprocessed_data",
        "predict_horizon": 1,
    },
}

TARGET_LABELS = {
    "temperature": "大棚温度 (°C)",
    "humidity": "大棚湿度 (%)",
    "temperature_outdoor": "户外温度 (°C)",
    "humidity_outdoor": "户外湿度 (%)",
    "light_intensity": "光照强度 (lux)",
    "wind_direction": "风向 (°)",
    "wind_speed": "风速 (m/s)",
    "rainfall": "降雨量 (mm)",
}

FEATURE_GROUPS = {
    "lag_short": ["lag_1", "lag_2", "lag_4"],
    "lag_medium": ["lag_96", "lag_192"],
    "lag_long": ["lag_672"],
    "time": ["hour", "minute", "day_of_week", "month", "is_daytime", "hour_sin", "hour_cos"],
    "rolling_mean": ["rolling_mean_4", "rolling_mean_16"],
    "rolling_std": ["rolling_std_4"],
    "wind_transforms": ["wind_sin", "wind_cos"],
    "cross": ["temp_diff", "humidity_diff", "temp_humidity_interaction"],
    "rainfall_binary": ["rain_flag"],
    "night_marker": ["is_night"],
}