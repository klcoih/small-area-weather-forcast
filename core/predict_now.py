#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时预测脚本 —— 加载已保存的 XGBoost 模型，基于最新数据预测下一个时间步

用法:
  python predict_now.py                        # 预测全部 8 个目标
  python predict_now.py --target outdoor_temperature  # 只预测指定目标
  python predict_now.py --steps 4              # 预测未来 4 步（迭代预测）
  python predict_now.py --json                 # 输出 JSON 格式
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'models')
DATA_DIR = os.path.join(CURRENT_DIR, '..', '天气数据')

GREENHOUSE_CSV = os.path.join(DATA_DIR, '温湿度数据', '温湿度数据.csv')
OUTDOOR_CSV = os.path.join(DATA_DIR, '宣威市尚营种气象', '宣威市尚营种气象.csv')

GREENHOUSE_TARGETS = ['greenhouse_temperature', 'greenhouse_humidity']
OUTDOOR_TARGETS = [
    'outdoor_temperature', 'outdoor_humidity',
    'light_intensity', 'wind_direction', 'wind_speed', 'rainfall'
]
ALL_TARGETS = GREENHOUSE_TARGETS + OUTDOOR_TARGETS

TARGET_LABELS = {
    'greenhouse_temperature': '大棚温度',
    'greenhouse_humidity': '大棚湿度',
    'outdoor_temperature': '室外温度',
    'outdoor_humidity': '室外湿度',
    'light_intensity': '光照强度',
    'wind_direction': '风向',
    'wind_speed': '风速',
    'rainfall': '降雨量',
}

TARGET_UNITS = {
    'greenhouse_temperature': '°C',
    'greenhouse_humidity': '%',
    'outdoor_temperature': '°C',
    'outdoor_humidity': '%',
    'light_intensity': 'lux',
    'wind_direction': '°',
    'wind_speed': 'm/s',
    'rainfall': 'mm',
}


class ModelLoader:
    """加载已保存的模型"""

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}

    def load_all(self):
        for fname in sorted(os.listdir(self.model_dir)):
            if not fname.endswith('.pkl'):
                continue
            target_name = fname.replace('.pkl', '')
            path = os.path.join(self.model_dir, fname)
            data = joblib.load(path)
            self.models[target_name] = data
        logger.info(f"已加载 {len(self.models)} 个模型: {list(self.models.keys())}")

    def load_one(self, target_name):
        path = os.path.join(self.model_dir, f'{target_name}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        self.models[target_name] = joblib.load(path)

    def get_meta(self, target_name):
        if target_name not in self.models:
            raise KeyError(f"模型未加载: {target_name}")
        return self.models[target_name].get('meta', {})

    def get_feature_columns(self, target_name):
        return self.get_meta(target_name).get('feature_columns', [])


class FeatureProvider:
    """特征提供器 —— 复用训练时的预处理管线生成特征"""

    def __init__(self):
        from core.data_preprocessing import PreprocessingPipeline
        self.pipeline = PreprocessingPipeline(output_dir=None)

    def prepare(self):
        from core.greenhouse_data_loader import GreenhouseDataLoader
        from core.outdoor_weather_loader import OutdoorWeatherLoader

        logger.info("加载原始数据...")
        gh_loader = GreenhouseDataLoader(GREENHOUSE_CSV)
        gh_loader.load()
        gh_loader.check_quality()
        gh_df = gh_loader.standardize()
        logger.info(f"大棚原始数据: {len(gh_df)} 条")

        ow_loader = OutdoorWeatherLoader(OUTDOOR_CSV)
        ow_loader.load()
        ow_loader.check_quality()
        ow_df = ow_loader.standardize()
        logger.info(f"室外原始数据: {len(ow_df)} 条")

        logger.info("运行特征工程管线...")
        self.pipeline.run(greenhouse_df=gh_df, outdoor_df=ow_df)

        self.gh_last = self.pipeline.greenhouse_df.iloc[-1] if self.pipeline.greenhouse_df is not None else None
        self.ow_last = self.pipeline.outdoor_df.iloc[-1] if self.pipeline.outdoor_df is not None else None

        if self.gh_last is not None:
            gh_ts = self.pipeline.greenhouse_df['timestamp'].iloc[-1]
            logger.info(f"大棚特征: {len(self.pipeline.greenhouse_df)} 行, "
                        f"{len(self.pipeline.greenhouse_df.columns)} 列, "
                        f"最新时间: {gh_ts}")

        if self.ow_last is not None:
            ow_ts = self.pipeline.outdoor_df['timestamp'].iloc[-1]
            logger.info(f"室外特征: {len(self.pipeline.outdoor_df)} 行, "
                        f"{len(self.pipeline.outdoor_df.columns)} 列, "
                        f"最新时间: {ow_ts}")

    def get_features(self, target_name):
        if target_name in GREENHOUSE_TARGETS:
            return self.gh_last
        else:
            return self.ow_last


class Predictor:
    """预测器 —— 加载模型并对特征进行预测"""

    def __init__(self, loader: ModelLoader, provider: FeatureProvider):
        self.loader = loader
        self.provider = provider

    def predict_single(self, target_name):
        if target_name not in self.loader.models:
            raise KeyError(f"模型未加载: {target_name}")

        model_data = self.loader.models[target_name]
        meta = model_data.get('meta', {})
        target_type = meta.get('target_type', 'regression')
        feature_cols = meta.get('feature_columns', [])

        features = self.provider.get_features(target_name)
        if features is None:
            return {'target': target_name, 'error': '特征数据不可用'}

        X = pd.DataFrame([features[feature_cols].values], columns=feature_cols)

        result = {
            'target': target_name,
            'label': TARGET_LABELS.get(target_name, target_name),
            'unit': TARGET_UNITS.get(target_name, ''),
            'type': target_type,
        }

        if target_type == 'regression':
            pred = float(model_data['model'].predict(X)[0])
            result['prediction'] = round(pred, 4)

        elif target_type == 'angular_regression':
            sin_pred = float(model_data['model_sin'].predict(X)[0])
            cos_pred = float(model_data['model_cos'].predict(X)[0])
            angle = float(np.degrees(np.arctan2(sin_pred, cos_pred)) % 360)
            result['prediction'] = round(angle, 2)
            result['wind_sin'] = round(sin_pred, 4)
            result['wind_cos'] = round(cos_pred, 4)

        elif target_type == 'two_stage':
            rain_prob = float(model_data['cls_model'].predict_proba(X)[0, 1])
            rain_flag = int(rain_prob >= 0.5)
            result['rain_probability'] = round(rain_prob, 4)
            result['rain_flag'] = rain_flag
            if rain_flag and model_data.get('reg_model') is not None:
                rain_amount = float(model_data['reg_model'].predict(X)[0])
                result['prediction'] = round(max(0, rain_amount), 4)
            else:
                result['prediction'] = 0.0

        return result

    def predict_all(self, targets=None):
        if targets is None:
            targets = ALL_TARGETS
        results = []
        for t in targets:
            try:
                results.append(self.predict_single(t))
            except Exception as e:
                logger.error(f"预测 {t} 失败: {e}")
                results.append({'target': t, 'error': str(e)})
        return results


def format_output(results, fmt='table'):
    if fmt == 'json':
        return json.dumps(results, ensure_ascii=False, indent=2)

    lines = []
    lines.append("")
    lines.append("=" * 58)
    lines.append("  实时气象预测结果")
    lines.append("=" * 58)

    gh_results = [r for r in results if r['target'] in GREENHOUSE_TARGETS]
    ow_results = [r for r in results if r['target'] in OUTDOOR_TARGETS]

    if gh_results:
        lines.append("")
        lines.append("  [大棚预测]")
        lines.append("  " + "-" * 54)
        for r in gh_results:
            if 'error' in r:
                lines.append(f"    {r['label']:12s}  ERROR: {r['error']}")
            else:
                lines.append(f"    {r['label']:12s}  {r['prediction']:>10.4f} {r['unit']}")

    if ow_results:
        lines.append("")
        lines.append("  [室外预测]")
        lines.append("  " + "-" * 54)
        for r in ow_results:
            if 'error' in r:
                lines.append(f"    {r['label']:12s}  ERROR: {r['error']}")
            elif r['type'] == 'two_stage':
                rain_text = "有雨" if r.get('rain_flag') else "无雨"
                lines.append(f"    {r['label']:12s}  {rain_text} (概率={r.get('rain_probability', 0):.4f}), "
                             f"降雨量={r.get('prediction', 0):.4f} {r['unit']}")
            else:
                lines.append(f"    {r['label']:12s}  {r['prediction']:>10.4f} {r['unit']}")

    lines.append("")
    lines.append("=" * 58)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='实时气象预测')
    parser.add_argument('--target', type=str, help='只预测指定目标')
    parser.add_argument('--json', action='store_true', help='JSON 格式输出')
    args = parser.parse_args()

    logger.info("初始化实时预测系统...")

    loader = ModelLoader()
    provider = FeatureProvider()

    loader.load_all()

    targets_to_predict = [args.target] if args.target else ALL_TARGETS
    unavailable = [t for t in targets_to_predict if t not in loader.models]
    if unavailable:
        logger.error(f"以下模型未找到: {unavailable}")
        return 1

    provider.prepare()

    predictor = Predictor(loader, provider)
    results = predictor.predict_all(targets_to_predict)

    output = format_output(results, fmt='json' if args.json else 'table')
    print(output)

    return 0


if __name__ == '__main__':
    sys.exit(main())