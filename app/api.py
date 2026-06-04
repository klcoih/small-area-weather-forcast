#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气象预测系统 REST API —— 基于 FastAPI

启动:
  python api.py
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

端点:
  GET  /api/health              健康检查
  GET  /api/models               模型列表
  GET  /api/predict              预测全部目标
  GET  /api/predict/{target}     预测指定目标
"""

import os
import sys
import time
import json
import logging
import webbrowser
import threading
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, '..'))

# ──────────────────────────────────────────────
# 数据模型
# ──────────────────────────────────────────────

class TargetInfo(BaseModel):
    target: str
    label: str
    unit: str
    type: str
    features: int
    samples: int

class PredictionItem(BaseModel):
    target: str
    label: str
    unit: str
    type: str
    prediction: float
    rain_probability: Optional[float] = None
    rain_flag: Optional[int] = None
    next15_prediction: Optional[float] = None
    next15_rain_probability: Optional[float] = None

class PredictResponse(BaseModel):
    timestamp: str
    predict_timestamp: str
    greenhouse_latest: Optional[str] = None
    outdoor_latest: Optional[str] = None
    predictions: list[PredictionItem]

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    uptime: float

class ForecastStepItem(BaseModel):
    step: int
    timestamp: str
    predictions: dict

class ForecastTimeline(BaseModel):
    latest_timestamp: Optional[str] = None
    timeline: list[ForecastStepItem]

class ForecastResponse(BaseModel):
    timestamp: str
    greenhouse: Optional[ForecastTimeline] = None
    outdoor: Optional[ForecastTimeline] = None
    steps: int
    step_minutes: int
    summary: dict

class UploadStatus(BaseModel):
    status: str
    message: str
    greenhouse_count: int
    outdoor_count: int

# ──────────────────────────────────────────────
# 全局状态（延迟加载）
# ──────────────────────────────────────────────

_loader = None
_provider = None
_predictor = None
_start_time = None
_raw_gh_df = None
_raw_ow_df = None

GREENHOUSE_TARGETS = ['greenhouse_temperature', 'greenhouse_humidity']
OUTDOOR_TARGETS = [
    'outdoor_temperature', 'outdoor_humidity',
    'light_intensity', 'wind_direction', 'wind_speed', 'rainfall'
]
ALL_TARGETS = GREENHOUSE_TARGETS + OUTDOOR_TARGETS

TARGET_LABELS = {
    'greenhouse_temperature': '大棚温度', 'greenhouse_humidity': '大棚湿度',
    'outdoor_temperature': '室外温度', 'outdoor_humidity': '室外湿度',
    'light_intensity': '光照强度', 'wind_direction': '风向',
    'wind_speed': '风速', 'rainfall': '降雨量',
}

TARGET_UNITS = {
    'greenhouse_temperature': '°C', 'greenhouse_humidity': '%',
    'outdoor_temperature': '°C', 'outdoor_humidity': '%',
    'light_intensity': 'lux', 'wind_direction': '°',
    'wind_speed': 'm/s', 'rainfall': 'mm',
}


def init_system():
    global _loader, _provider, _predictor, _start_time, _raw_gh_df, _raw_ow_df
    from core.greenhouse_data_loader import GreenhouseDataLoader
    from core.outdoor_weather_loader import OutdoorWeatherLoader
    from core.predict_now import ModelLoader, FeatureProvider, Predictor

    logger.info("正在初始化预测系统...")
    _loader = ModelLoader()
    _loader.load_all()

    gh_loader = GreenhouseDataLoader(os.path.join(CURRENT_DIR, '..', 'data', '天气数据', '温湿度数据', '温湿度数据.csv'))
    gh_loader.load(); gh_loader.check_quality()
    _raw_gh_df = gh_loader.standardize()

    ow_loader = OutdoorWeatherLoader(os.path.join(CURRENT_DIR, '..', 'data', '天气数据', '宣威市尚营种气象', '宣威市尚营种气象.csv'))
    ow_loader.load(); ow_loader.check_quality()
    _raw_ow_df = ow_loader.standardize()

    _provider = FeatureProvider()
    _provider.pipeline.run(greenhouse_df=_raw_gh_df, outdoor_df=_raw_ow_df)
    _provider.gh_last = _provider.pipeline.greenhouse_df.iloc[-1] if _provider.pipeline.greenhouse_df is not None else None
    _provider.ow_last = _provider.pipeline.outdoor_df.iloc[-1] if _provider.pipeline.outdoor_df is not None else None

    _predictor = Predictor(_loader, _provider)
    _start_time = time.time()
    logger.info(f"初始化完成，已加载 {len(_loader.models)} 个模型")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_system()
    print()
    print("  " + "=" * 52)
    print("   小区域气象预测系统已启动")
    print("   " + "-" * 48)
    print("   仪表盘:   http://127.0.0.1:8000/")
    print("   API文档:  http://127.0.0.1:8000/docs")
    print("   健康检查: http://127.0.0.1:8000/api/health")
    print("  " + "=" * 52)
    print()
    yield


app = FastAPI(
    title="小区域气象预测系统 API",
    description="大棚与室外环境 15 分钟级实时预测服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# API 端点
# ──────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["系统"])
def health_check():
    return HealthResponse(
        status="ok",
        models_loaded=len(_loader.models) if _loader else 0,
        uptime=round(time.time() - _start_time, 1) if _start_time else 0,
    )


@app.get("/api/models", tags=["系统"])
def list_models():
    models = []
    for target_name in ALL_TARGETS:
        if target_name not in _loader.models:
            continue
        meta = _loader.get_meta(target_name)
        models.append(TargetInfo(
            target=target_name,
            label=TARGET_LABELS.get(target_name, target_name),
            unit=TARGET_UNITS.get(target_name, ''),
            type=meta.get('target_type', 'regression'),
            features=meta.get('n_features', 0),
            samples=meta.get('n_samples', 0),
        ))
    return {"models": models}


@app.get("/api/predict", response_model=PredictResponse, tags=["预测"])
def predict_all():
    results = _predictor.predict_all(ALL_TARGETS)

    gh_latest_ts = None
    ow_latest_ts = None
    if _raw_gh_df is not None:
        gh_latest_ts = _raw_gh_df['timestamp'].max()
    if _raw_ow_df is not None:
        ow_latest_ts = _raw_ow_df['timestamp'].max()

    # 快速计算下15分钟（第2步）的预测值
    next15_preds = {}
    try:
        from core.forecast_engine import RecursiveForecaster
        forecaster = RecursiveForecaster(_loader)
        result = forecaster.forecast_to_timeline(
            greenhouse_df=_provider.pipeline.greenhouse_df,
            outdoor_df=_provider.pipeline.outdoor_df,
            steps=2,
            latest_gh_timestamp=gh_latest_ts,
            latest_ow_timestamp=ow_latest_ts,
        )
        # 分别处理大棚和室外的下15分钟预测
        for key in ['greenhouse', 'outdoor']:
            if result.get(key) and result[key].get('timeline') and len(result[key]['timeline']) >= 2:
                step2 = result[key]['timeline'][1]
                if step2.get('predictions'):
                    for target_key, value in step2['predictions'].items():
                        next15_preds[target_key] = value
    except Exception as e:
        logger.warning(f"计算下15分钟预测失败: {e}")

    items = []
    for r in results:
        target_name = r['target']
        next15_val = next15_preds.get(target_name)

        # 映射: outdoor_temperature → outdoor_temperature (key in predictions)
        next15_rain_prob = None
        if target_name == 'rainfall' and 'rainfall' in next15_preds:
            # 对于降雨量，需要重新计算概率
            pass

        item = PredictionItem(
            target=r['target'],
            label=r['label'],
            unit=r['unit'],
            type=r['type'],
            prediction=r.get('prediction', 0),
            rain_probability=r.get('rain_probability'),
            rain_flag=r.get('rain_flag'),
            next15_prediction=next15_val if next15_val is not None else r.get('prediction', 0),
            next15_rain_probability=r.get('rain_probability'),
        )
        items.append(item)

    gh_ts = None
    ow_ts = None
    if _provider.pipeline.greenhouse_df is not None:
        gh_ts = str(_provider.pipeline.greenhouse_df['timestamp'].iloc[-1])
    if _provider.pipeline.outdoor_df is not None:
        ow_ts = str(_provider.pipeline.outdoor_df['timestamp'].iloc[-1])

    return PredictResponse(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        predict_timestamp=ow_ts or gh_ts or '',
        greenhouse_latest=gh_ts,
        outdoor_latest=ow_ts,
        predictions=items,
    )


@app.get("/api/predict/{target_name}", response_model=PredictResponse, tags=["预测"])
def predict_single(target_name: str):
    if target_name not in ALL_TARGETS:
        raise HTTPException(status_code=404, detail=f"未知目标: {target_name}")
    if target_name not in _loader.models:
        raise HTTPException(status_code=503, detail=f"模型未加载: {target_name}")

    result = _predictor.predict_single(target_name)
    items = [PredictionItem(
        target=result['target'],
        label=result['label'],
        unit=result['unit'],
        type=result['type'],
        prediction=result.get('prediction', 0),
        rain_probability=result.get('rain_probability'),
        rain_flag=result.get('rain_flag'),
    )]

    gh_ts = None
    ow_ts = None
    if _provider.pipeline.greenhouse_df is not None:
        gh_ts = str(_provider.pipeline.greenhouse_df['timestamp'].iloc[-1])
    if _provider.pipeline.outdoor_df is not None:
        ow_ts = str(_provider.pipeline.outdoor_df['timestamp'].iloc[-1])

    return PredictResponse(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        predict_timestamp=ow_ts or gh_ts or '',
        greenhouse_latest=gh_ts,
        outdoor_latest=ow_ts,
        predictions=items,
    )


def reload_system():
    """重新加载管线（上传新数据后调用）"""
    global _provider, _predictor, _raw_gh_df, _raw_ow_df
    from core.predict_now import FeatureProvider, Predictor

    logger.info("重新加载数据管线...")
    _provider = FeatureProvider()
    _provider.pipeline.run(greenhouse_df=_raw_gh_df, outdoor_df=_raw_ow_df)
    _provider.gh_last = _provider.pipeline.greenhouse_df.iloc[-1] if _provider.pipeline.greenhouse_df is not None else None
    _provider.ow_last = _provider.pipeline.outdoor_df.iloc[-1] if _provider.pipeline.outdoor_df is not None else None

    _predictor = Predictor(_loader, _provider)
    logger.info("管线重新加载完成")


def _parse_timestamp(ts_str):
    """解析多种时间格式"""
    formats = [
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(str(ts_str).strip(), fmt)
        except ValueError:
            continue
    return pd.to_datetime(ts_str)


@app.post("/api/upload", response_model=UploadStatus, tags=["数据"])
async def upload_data(file: UploadFile = File(...)):
    """上传传感器数据（CSV 或 JSON），自动识别大棚/室外格式并追加到管线"""
    global _raw_gh_df, _raw_ow_df

    content = await file.read()
    text = content.decode('utf-8-sig')

    gh_count = 0
    ow_count = 0

    try:
        data = json.loads(text)

        if 'greenhouse' in data and data['greenhouse']:
            gh_new = pd.DataFrame(data['greenhouse'])
            gh_new = _standardize_upload_columns(gh_new, 'greenhouse')
            _raw_gh_df = pd.concat([_raw_gh_df, gh_new], ignore_index=True)
            gh_count = len(gh_new)
            logger.info(f"JSON: 追加 {gh_count} 条大棚数据")

        if 'outdoor' in data and data['outdoor']:
            ow_new = pd.DataFrame(data['outdoor'])
            ow_new = _standardize_upload_columns(ow_new, 'outdoor')
            _raw_ow_df = pd.concat([_raw_ow_df, ow_new], ignore_index=True)
            ow_count = len(ow_new)
            logger.info(f"JSON: 追加 {ow_count} 条室外数据")

    except (json.JSONDecodeError, ValueError):
        from io import StringIO
        df = pd.read_csv(StringIO(text))
        df = _standardize_upload_columns(df, None)
        loc = df['location'].iloc[0] if 'location' in df.columns else None

        if loc == 'greenhouse' or ('temperature' in df.columns and 'humidity' in df.columns and 'temperature_outdoor' not in df.columns):
            _raw_gh_df = pd.concat([_raw_gh_df, df.drop(columns=['location'], errors='ignore')], ignore_index=True)
            gh_count = len(df)
            logger.info(f"CSV: 追加 {gh_count} 条大棚数据")
        elif loc == 'outdoor' or 'temperature_outdoor' in df.columns or 'light_intensity' in df.columns:
            _raw_ow_df = pd.concat([_raw_ow_df, df.drop(columns=['location'], errors='ignore')], ignore_index=True)
            ow_count = len(df)
            logger.info(f"CSV: 追加 {ow_count} 条室外数据")
        else:
            raise HTTPException(400, "无法识别数据格式，支持:\n大棚: 时间,温度,湿度\n室外: 时间,温度,湿度,光照强度,风向,风速,降雨量")

    reload_system()

    return UploadStatus(
        status="ok",
        message=f"已追加并重新生成特征: 大棚 +{gh_count}条, 室外 +{ow_count}条",
        greenhouse_count=gh_count,
        outdoor_count=ow_count,
    )


# 列名映射表：支持中文/英文列名自动转换
_GH_COLUMN_MAP = {
    '时间': 'timestamp', 'timestamp': 'timestamp',
    '温度': 'temperature', 'temperature': 'temperature',
    '湿度': 'humidity', 'humidity': 'humidity',
    '序号': None, 'id': None, 'index': None,
}

_OW_COLUMN_MAP = {
    '时间': 'timestamp', 'timestamp': 'timestamp',
    '温度': 'temperature_outdoor', 'temperature_outdoor': 'temperature_outdoor',
    '湿度': 'humidity_outdoor', 'humidity_outdoor': 'humidity_outdoor',
    '光照强度': 'light_intensity', 'light_intensity': 'light_intensity',
    '风向': 'wind_direction', 'wind_direction': 'wind_direction',
    '风速': 'wind_speed', 'wind_speed': 'wind_speed',
    '降雨量': 'rainfall', 'rainfall': 'rainfall',
    '序号': None, 'id': None, 'index': None,
}


def _standardize_upload_columns(df: pd.DataFrame, force_type: str = None) -> pd.DataFrame:
    """将上传的 DataFrame 列名标准化为内部格式，自动检测大棚/室外"""
    df = df.copy()

    # 自动检测：室外有光照强度/风向/风速/降雨量
    ow_indicators = {'光照强度', '风向', '风速', '降雨量', 'light_intensity', 'wind_direction', 'wind_speed', 'rainfall'}
    gh_indicators = {'温度', '湿度', 'temperature', 'humidity'}

    has_ow = bool(ow_indicators & set(df.columns))
    has_gh = bool(gh_indicators & set(df.columns)) and not has_ow

    if force_type == 'greenhouse':
        col_map = _GH_COLUMN_MAP
        df['location'] = 'greenhouse'
    elif force_type == 'outdoor':
        col_map = _OW_COLUMN_MAP
        df['location'] = 'outdoor'
    elif has_ow:
        col_map = _OW_COLUMN_MAP
        df['location'] = 'outdoor'
    elif has_gh:
        col_map = _GH_COLUMN_MAP
        df['location'] = 'greenhouse'
    else:
        raise HTTPException(400, "无法识别数据格式，请检查列名")

    # 重命名
    rename_map = {}
    for col in df.columns:
        if col in col_map:
            if col_map[col] is not None and col != col_map[col]:
                rename_map[col] = col_map[col]
        elif col == 'location':
            pass
        else:
            logger.warning(f"忽略未知列: {col}")

    if rename_map:
        df = df.rename(columns=rename_map)

    # 删除映射为 None 的列
    drop_cols = [col for col in df.columns if col in col_map and col_map[col] is None]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 确保 timestamp 列存在
    if 'timestamp' not in df.columns:
        raise HTTPException(400, "数据缺少 timestamp/时间 列")

    # 解析时间
    df['timestamp'] = df['timestamp'].apply(_parse_timestamp)

    return df


@app.post("/api/forecast", response_model=ForecastResponse, tags=["预测"])
def forecast(steps: int = 96):
    """递归预测未来 N 步（默认 96 = 24小时）
    室外和大棚数据分别从各自的最新时间开始预测"""
    if steps < 1 or steps > 672:
        raise HTTPException(status_code=400, detail="步数需在 1-672 之间")

    from core.forecast_engine import RecursiveForecaster, build_forecast_summary

    forecaster = RecursiveForecaster(_loader)

    # 使用原始数据的最新时间作为预测起点
    gh_latest_ts = None
    ow_latest_ts = None
    if _raw_gh_df is not None:
        gh_latest_ts = _raw_gh_df['timestamp'].max()
    if _raw_ow_df is not None:
        ow_latest_ts = _raw_ow_df['timestamp'].max()

    result = forecaster.forecast_to_timeline(
        greenhouse_df=_provider.pipeline.greenhouse_df,
        outdoor_df=_provider.pipeline.outdoor_df,
        steps=steps,
        latest_gh_timestamp=gh_latest_ts,
        latest_ow_timestamp=ow_latest_ts,
    )

    # 分别构建室外和大棚的时间线
    greenhouse_timeline = None
    outdoor_timeline = None

    if 'greenhouse' in result and result['greenhouse']:
        greenhouse_timeline = ForecastTimeline(
            latest_timestamp=result['greenhouse']['latest_timestamp'],
            timeline=[ForecastStepItem(**t) for t in result['greenhouse']['timeline']]
        )

    if 'outdoor' in result and result['outdoor']:
        outdoor_timeline = ForecastTimeline(
            latest_timestamp=result['outdoor']['latest_timestamp'],
            timeline=[ForecastStepItem(**t) for t in result['outdoor']['timeline']]
        )

    # 合并所有预测构建摘要
    all_timeline = []
    if greenhouse_timeline:
        all_timeline.extend(greenhouse_timeline.timeline)
    if outdoor_timeline:
        all_timeline.extend(outdoor_timeline.timeline)

    summary = build_forecast_summary(all_timeline)

    return ForecastResponse(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        greenhouse=greenhouse_timeline,
        outdoor=outdoor_timeline,
        steps=steps,
        step_minutes=15,
        summary=summary,
    )


@app.get("/api/forecast", response_model=ForecastResponse, tags=["预测"])
def forecast_get(steps: int = 96):
    """GET 方式递归预测（浏览器友好）"""
    return forecast(steps=steps)


@app.get("/api/history", tags=["数据"])
def get_history(group: str = "all", hours: int = 24):
    """
    获取历史气象数据用于可视化

    Args:
        group: outdoor | greenhouse | all
        hours: 回溯小时数（默认 24，最大 720 = 30天）
    """
    hours = max(1, min(hours, 720))
    result = {}

    if group in ("greenhouse", "all") and _raw_gh_df is not None:
        gh = _raw_gh_df.copy()
        cutoff = gh['timestamp'].max() - timedelta(hours=hours)
        gh = gh[gh['timestamp'] >= cutoff].sort_values('timestamp')
        # Down-sample if too many points (> 1000)
        if len(gh) > 1000:
            gh = gh.iloc[::len(gh) // 1000]
        result['greenhouse'] = {
            'timestamps': [str(t) for t in gh['timestamp'].tolist()],
            'temperature': [round(float(v), 2) for v in gh['temperature'].tolist()],
            'humidity': [round(float(v), 2) for v in gh['humidity'].tolist()],
            'count': len(gh),
            'hours': hours,
        }

    if group in ("outdoor", "all") and _raw_ow_df is not None:
        ow = _raw_ow_df.copy()
        cutoff = ow['timestamp'].max() - timedelta(hours=hours)
        ow = ow[ow['timestamp'] >= cutoff].sort_values('timestamp')
        if len(ow) > 2000:
            ow = ow.iloc[::len(ow) // 2000]
        result['outdoor'] = {
            'timestamps': [str(t) for t in ow['timestamp'].tolist()],
            'temperature_outdoor': [round(float(v), 2) for v in ow['temperature_outdoor'].tolist()],
            'humidity_outdoor': [round(float(v), 2) for v in ow['humidity_outdoor'].tolist()],
            'light_intensity': [round(float(v), 2) for v in ow['light_intensity'].tolist()],
            'wind_direction': [round(float(v), 2) for v in ow['wind_direction'].tolist()],
            'wind_speed': [round(float(v), 2) for v in ow['wind_speed'].tolist()],
            'rainfall': [round(float(v), 4) for v in ow['rainfall'].tolist()],
            'count': len(ow),
            'hours': hours,
        }

    return result


@app.get("/", response_class=HTMLResponse)
def index():
    dashboard_path = os.path.join(CURRENT_DIR, 'dashboard.html')
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h2>API running. Visit <a href='/docs'>/docs</a> for Swagger UI.</h2>"


if __name__ == '__main__':
    import uvicorn

    def _open_browser():
        time.sleep(3)
        webbrowser.open('http://127.0.0.1:8000/')

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)