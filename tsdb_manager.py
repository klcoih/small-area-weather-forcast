#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序数据库管理模块 —— 基于 InfluxDB 2.x

数据特点：
  - 时间间隔：15分钟（96条/天/目标）
  - 预测目标：8个
  - 预估数据量：每年约35万条记录

Schema 设计（统一表方案）：
  Measurement: environment_data
    Tags:
      - location: greenhouse | outdoor
      - metric:  temperature | humidity | temperature_outdoor |
                 humidity_outdoor | light_intensity |
                 wind_direction | wind_speed | rainfall
    Fields:
      - value:  float  测量值
      - quality: int   数据质量标记（0=正常, 1=异常, 2=插值）
    Timestamp: 对齐到15分钟整点

查询能力：
  - 按时间范围、location、metric 过滤
  - 降采样：15min → 1h → 1d
  - 聚合：mean / min / max / last

使用示例:
    from tsdb_manager import TimeSeriesDB

    db = TimeSeriesDB(url="http://localhost:8086", token="my-token",
                       org="my-org", bucket="weather_data")

    db.initialize()

    points = [{"time": datetime.now(), "location": "greenhouse",
               "metric": "temperature", "value": 22.5, "quality": 0}]
    db.write_batch(points)

    df = db.query_range("-1h", metric="temperature", location="greenhouse")
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Generator, Union

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def round_to_15min(dt: datetime) -> datetime:
    """
    将任意时间对齐到最近的15分钟整点

    Args:
        dt: 任意datetime对象

    Returns:
        datetime: 对齐后的时间（秒和微秒置零）

    Examples:
        12:07:10 → 12:00:00
        12:08:00 → 12:15:00
        12:16:39 → 12:15:00
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    minute = dt.minute
    aligned_minute = (minute // 15) * 15
    return dt.replace(minute=aligned_minute, second=0, microsecond=0)


def _try_import_influxdb():
    """延迟导入 influxdb_client，错误时给出友好提示"""
    try:
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
        from influxdb_client.client.write.retry import WritesRetry
        from influxdb_client.rest import ApiException
        return InfluxDBClient, SYNCHRONOUS, ASYNCHRONOUS, WritesRetry, ApiException
    except ImportError:
        raise ImportError(
            "请安装 influxdb-client:  pip install influxdb-client"
        )


def _try_import_influxdb_query():
    """延迟导入查询API"""
    try:
        from influxdb_client.client.flux_table import FluxStructureEncoder
        return FluxStructureEncoder
    except ImportError:
        return None


METRIC_META = {
    'temperature':         {'location': 'greenhouse', 'unit': '°C',       'label': '大棚温度'},
    'humidity':            {'location': 'greenhouse', 'unit': '%',        'label': '大棚湿度'},
    'temperature_outdoor': {'location': 'outdoor',    'unit': '°C',       'label': '户外温度'},
    'humidity_outdoor':    {'location': 'outdoor',    'unit': '%',        'label': '户外湿度'},
    'light_intensity':     {'location': 'outdoor',    'unit': 'lux',      'label': '光照强度'},
    'wind_direction':      {'location': 'outdoor',    'unit': '°',        'label': '风向'},
    'wind_speed':          {'location': 'outdoor',    'unit': 'm/s',      'label': '风速'},
    'rainfall':            {'location': 'outdoor',    'unit': 'mm',       'label': '降雨量'},
}

LOCATION_META = {
    'greenhouse': {'metrics': ['temperature', 'humidity'],                  'label': '大棚'},
    'outdoor':    {'metrics': ['temperature_outdoor', 'humidity_outdoor',
                               'light_intensity', 'wind_direction',
                               'wind_speed', 'rainfall'],                  'label': '户外'},
}

VALID_METRICS = list(METRIC_META.keys())
VALID_LOCATIONS = list(LOCATION_META.keys())


class TimeSeriesDB:
    """
    InfluxDB 2.x 时序数据库管理器

    封装写入 / 查询 / 初始化 / 连接池管理
    """

    def __init__(self, url: str = "http://localhost:8086",
                 token: str = "",
                 org: str = "my-org",
                 bucket: str = "weather_data",
                 measurement: str = "environment_data",
                 timeout: int = 30000):
        """
        Args:
            url:     InfluxDB 地址
            token:   API Token
            org:     组织名称
            bucket:  桶名称
            measurement: 测量名称
            timeout: HTTP 超时毫秒
        """
        self._url = url
        self._token = token
        self._org = org
        self._bucket = bucket
        self._measurement = measurement
        self._timeout = timeout

        self._client = None
        self._write_api = None
        self._query_api = None
        self._delete_api = None

    # ── 连接管理 ─────────────────────────────────────────────

    def connect(self):
        """建立 InfluxDB 连接"""
        InfluxDBClient, _, _, _, _ = _try_import_influxdb()

        if self._client is not None:
            return

        self._client = InfluxDBClient(
            url=self._url,
            token=self._token,
            org=self._org,
            timeout=self._timeout,
            enable_gzip=True,
        )
        self._write_api = self._client.write_api()
        self._query_api = self._client.query_api()
        self._delete_api = self._client.delete_api()

        logger.info(f"已连接 InfluxDB: {self._url} (org={self._org}, bucket={self._bucket})")

    def close(self):
        """关闭连接"""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._write_api = None
            self._query_api = None
            self._delete_api = None
            logger.info("InfluxDB 连接已关闭")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def ping(self) -> bool:
        """健康检查"""
        try:
            if self._client is None:
                self.connect()
            self._client.ping()
            return True
        except Exception as e:
            logger.error(f"InfluxDB ping 失败: {e}")
            return False

    # ── 初始化 ───────────────────────────────────────────────

    def initialize(self):
        """
        初始化 Bucket（如果不存在则创建）

        Returns:
            bool: 是否成功
        """
        InfluxDBClient, _, _, _, ApiException = _try_import_influxdb()
        self.connect()

        try:
            buckets_api = self._client.buckets_api()
            existing = buckets_api.find_bucket_by_name(self._bucket)
            if existing is None:
                buckets_api.create_bucket(bucket_name=self._bucket, org=self._org)
                logger.info(f"Bucket '{self._bucket}' 创建成功")
            else:
                logger.info(f"Bucket '{self._bucket}' 已存在")
            return True
        except ApiException as e:
            logger.error(f"初始化 Bucket 失败: {e}")
            return False

    # ── 数据写入 ─────────────────────────────────────────────

    def _build_point(self, timestamp: datetime, location: str,
                     metric: str, value: float, quality: int = 0):
        """
        构建单条 InfluxDB Point

        Args:
            timestamp: 时间（自动对齐到15分钟）
            location:  greenhouse / outdoor
            metric:    指标名称
            value:     测量值
            quality:   0=正常, 1=异常, 2=插值

        Returns:
            influxdb_client.Point
        """
        InfluxDBClient, _, _, _, _ = _try_import_influxdb()
        from influxdb_client import Point

        aligned_time = round_to_15min(timestamp)

        return (
            Point(self._measurement)
            .tag("location", location)
            .tag("metric", metric)
            .field("value", float(value))
            .field("quality", int(quality))
            .time(aligned_time)
        )

    def write_point(self, timestamp: datetime, location: str,
                    metric: str, value: float, quality: int = 0) -> bool:
        """
        单条写入

        Args:
            timestamp: 数据时间
            location:  greenhouse / outdoor
            metric:    指标
            value:     测量值
            quality:   质量标记

        Returns:
            bool: 是否成功
        """
        _, SYNCHRONOUS, _, _, _ = _try_import_influxdb()
        self.connect()

        point = self._build_point(timestamp, location, metric, value, quality)
        try:
            self._write_api.write(
                bucket=self._bucket, org=self._org,
                record=point, write_precision='s'
            )
            return True
        except Exception as e:
            logger.error(f"单条写入失败 [{metric}@{location}]: {e}")
            return False

    def write_batch(self, points: List[Dict],
                    batch_size: int = 5000) -> int:
        """
        批量写入（历史数据导入）

        Args:
            points: 字典列表，每条格式:
                    {"time": datetime, "location": str, "metric": str,
                     "value": float, "quality": int}
            batch_size: 每批次大小

        Returns:
            int: 成功写入的条数
        """
        _, SYNCHRONOUS, _, _, _ = _try_import_influxdb()
        self.connect()

        if not points:
            return 0

        influx_points = []
        skipped = 0
        for p in points:
            loc = p.get("location", "")
            metric = p.get("metric", "")
            if loc not in VALID_LOCATIONS or metric not in VALID_METRICS:
                logger.warning(f"跳过无效记录: location={loc}, metric={metric}")
                skipped += 1
                continue
            influx_points.append(
                self._build_point(
                    p["time"], loc, metric,
                    p.get("value", 0.0),
                    p.get("quality", 0)
                )
            )

        total = len(influx_points)
        success = 0
        for i in range(0, total, batch_size):
            chunk = influx_points[i:i + batch_size]
            try:
                self._write_api.write(
                    bucket=self._bucket, org=self._org,
                    record=chunk, write_precision='s'
                )
                success += len(chunk)
            except Exception as e:
                logger.error(f"批量写入失败 (batch {i // batch_size}): {e}")

        logger.info(f"批量写入完成: 成功 {success}/{total} 条, 跳过 {skipped} 条")
        return success

    def write_dataframe(self, df: pd.DataFrame,
                        time_col: str = 'timestamp',
                        location_col: str = 'location') -> int:
        """
        从 DataFrame 导入数据

        支持两种 DataFrame 格式：

        格式A - 宽表（每个指标一列）：
          timestamp | temperature | humidity | location
          → 自动展开为多条 Point

        格式B - 长表（metric 作为值）：
          timestamp | location | metric | value | quality

        Args:
            df:       DataFrame
            time_col: 时间列名
            location_col: 位置列名

        Returns:
            int: 写入条数
        """
        if 'metric' in df.columns and 'value' in df.columns:
            return self._write_long_df(df, time_col)

        return self._write_wide_df(df, time_col, location_col)

    def _write_long_df(self, df: pd.DataFrame, time_col: str) -> int:
        """写入长表格式 DataFrame"""
        points = []
        for _, row in df.iterrows():
            points.append({
                "time": row[time_col],
                "location": row.get("location", "outdoor"),
                "metric": row["metric"],
                "value": row.get("value", 0.0),
                "quality": row.get("quality", 0)
            })
        return self.write_batch(points)

    def _write_wide_df(self, df: pd.DataFrame, time_col: str,
                       location_col: str) -> int:
        """写入宽表格式 DataFrame，自动展开指标"""
        points = []
        for _, row in df.iterrows():
            t = row[time_col]
            loc = row.get(location_col, "outdoor")
            for col in df.columns:
                if col in (time_col, location_col):
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                metric = col
                if metric == 'temperature_outdoor' and loc == 'greenhouse':
                    metric = 'temperature'
                elif metric == 'humidity_outdoor' and loc == 'greenhouse':
                    metric = 'humidity'
                if metric not in VALID_METRICS:
                    continue
                points.append({
                    "time": t, "location": loc, "metric": metric,
                    "value": float(val), "quality": 0
                })
        return self.write_batch(points)

    def import_from_greenhouse(self, loader, quality: int = 0) -> int:
        """
        从 GreenhouseDataLoader 导入大棚数据

        Args:
            loader:  GreenhouseDataLoader 实例（已调用 standardize()）
            quality: 质量标记

        Returns:
            int: 写入条数
        """
        if loader.df_standardized is None:
            loader.standardize()

        df = loader.df_standardized
        points = []
        for _, row in df.iterrows():
            t = row['timestamp']
            for metric in ['temperature', 'humidity']:
                if metric in df.columns and pd.notna(row[metric]):
                    points.append({
                        "time": t, "location": "greenhouse",
                        "metric": metric,
                        "value": float(row[metric]),
                        "quality": quality
                    })
        return self.write_batch(points)

    def import_from_outdoor(self, loader, quality: int = 0) -> int:
        """
        从 OutdoorWeatherLoader 导入户外数据

        Args:
            loader:  OutdoorWeatherLoader 实例（已调用 standardize()）
            quality: 质量标记

        Returns:
            int: 写入条数
        """
        if loader.df_standardized is None:
            loader.standardize()

        df = loader.df_standardized
        outdoor_metrics = ['temperature_outdoor', 'humidity_outdoor',
                           'light_intensity', 'wind_direction',
                           'wind_speed', 'rainfall']
        points = []
        for _, row in df.iterrows():
            t = row['timestamp']
            for metric in outdoor_metrics:
                if metric in df.columns and pd.notna(row[metric]):
                    points.append({
                        "time": t, "location": "outdoor",
                        "metric": metric,
                        "value": float(row[metric]),
                        "quality": quality
                    })
        return self.write_batch(points)

    # ── 数据查询 ─────────────────────────────────────────────

    def query_range(self,
                    start: Union[str, datetime],
                    stop: Optional[Union[str, datetime]] = None,
                    metric: Optional[Union[str, List[str]]] = None,
                    location: Optional[Union[str, List[str]]] = None,
                    every: Optional[str] = None,
                    aggregate: str = "mean") -> pd.DataFrame:
        """
        按时间范围查询

        Args:
            start:     起始时间（"-1h", "-7d" 或 datetime）
            stop:      结束时间，默认 now()
            metric:    指标过滤（单个或多个）
            location:  位置过滤
            every:     降采样间隔 ("1h", "6h", "1d")，None 则返回原始粒度
            aggregate: 聚合函数 (mean/min/max/last)

        Returns:
            DataFrame: columns=[_time, location, metric, value, quality]
        """
        self.connect()

        filters = [f'r._measurement == "{self._measurement}"']

        if metric:
            metrics = [metric] if isinstance(metric, str) else metric
            metric_filter = " or ".join(f'r.metric == "{m}"' for m in metrics)
            filters.append(f"({metric_filter})")

        if location:
            locations = [location] if isinstance(location, str) else location
            loc_filter = " or ".join(f'r.location == "{l}"' for l in locations)
            filters.append(f"({loc_filter})")

        filter_str = " and ".join(filters)

        if every:
            flux = f'''
                from(bucket: "{self._bucket}")
                  |> range(start: {self._flux_time(start)}, stop: {self._flux_time(stop)})
                  |> filter(fn: (r) => {filter_str})
                  |> aggregateWindow(every: {every}, fn: {aggregate}, createEmpty: false)
                  |> pivot(rowKey: ["_time", "location", "metric"],
                           columnKey: ["_field"], valueColumn: "_value")
                  |> keep(columns: ["_time", "location", "metric", "value", "quality"])
            '''
        else:
            flux = f'''
                from(bucket: "{self._bucket}")
                  |> range(start: {self._flux_time(start)}, stop: {self._flux_time(stop)})
                  |> filter(fn: (r) => {filter_str})
                  |> pivot(rowKey: ["_time", "location", "metric"],
                           columnKey: ["_field"], valueColumn: "_value")
                  |> keep(columns: ["_time", "location", "metric", "value", "quality"])
            '''

        try:
            tables = self._query_api.query(flux, org=self._org)
            records = []
            for table in tables:
                for row in table.records:
                    records.append({
                        '_time': row['_time'],
                        'location': row['location'],
                        'metric': row['metric'],
                        'value': row['value'],
                        'quality': row.get('quality', 0)
                    })
            df = pd.DataFrame(records)
            if not df.empty:
                df['_time'] = pd.to_datetime(df['_time'])
                df = df.sort_values(['_time', 'location', 'metric']).reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return pd.DataFrame()

    def query_latest(self, metric: Optional[str] = None,
                     location: Optional[str] = None) -> pd.DataFrame:
        """
        查询最新一条数据

        Args:
            metric:   过滤指标
            location: 过滤位置

        Returns:
            DataFrame
        """
        self.connect()

        filters = [f'r._measurement == "{self._measurement}"']
        if metric:
            filters.append(f'r.metric == "{metric}"')
        if location:
            filters.append(f'r.location == "{location}"')
        filter_str = " and ".join(filters)

        flux = f'''
            from(bucket: "{self._bucket}")
              |> range(start: -30d)
              |> filter(fn: (r) => {filter_str})
              |> last()
              |> pivot(rowKey: ["_time", "location", "metric"],
                       columnKey: ["_field"], valueColumn: "_value")
              |> keep(columns: ["_time", "location", "metric", "value", "quality"])
        '''

        try:
            tables = self._query_api.query(flux, org=self._org)
            records = []
            for table in tables:
                for row in table.records:
                    records.append({
                        '_time': row['_time'],
                        'location': row['location'],
                        'metric': row['metric'],
                        'value': row['value'],
                        'quality': row.get('quality', 0)
                    })
            df = pd.DataFrame(records)
            if not df.empty:
                df['_time'] = pd.to_datetime(df['_time'])
            return df
        except Exception as e:
            logger.error(f"查询最新数据失败: {e}")
            return pd.DataFrame()

    def query_downsample(self, start: str = "-7d",
                         every: str = "1h",
                         aggregate: str = "mean") -> pd.DataFrame:
        """
        降采样查询便捷方法

        Args:
            start:     起始时间
            every:     降采样间隔 "1h" / "6h" / "1d"
            aggregate: mean / min / max / last

        Returns:
            DataFrame: 降采样后的数据
        """
        return self.query_range(start=start, every=every, aggregate=aggregate)

    def query_pivot_wide(self, start: str = "-1d",
                         location: Optional[str] = None) -> pd.DataFrame:
        """
        查询并转换为宽表（每个指标一列）

        Args:
            start:    起始时间
            location: 位置过滤

        Returns:
            DataFrame: columns=[_time, temperature, humidity, ...]
        """
        df = self.query_range(start=start, location=location)

        if df.empty:
            return df

        df_wide = df.pivot_table(
            index='_time', columns='metric',
            values='value', aggfunc='first'
        ).reset_index()

        df_wide.columns.name = None
        return df_wide

    # ── 统计查询 ─────────────────────────────────────────────

    def get_stats(self, start: str = "-1d",
                  metric: Optional[str] = None,
                  location: Optional[str] = None) -> Dict:
        """
        获取统计摘要（均值/最小/最大/标准差）

        Returns:
            dict: {metric: {mean, min, max, std}}
        """
        self.connect()

        filters = [f'r._measurement == "{self._measurement}"']
        if metric:
            filters.append(f'r.metric == "{metric}"')
        if location:
            filters.append(f'r.location == "{location}"')
        filter_str = " and ".join(filters)

        flux = f'''
            data = from(bucket: "{self._bucket}")
              |> range(start: {self._flux_time(start)})
              |> filter(fn: (r) => {filter_str})
              |> filter(fn: (r) => r._field == "value")

            data
              |> group(columns: ["location", "metric"])
              |> mean()
              |> set(key: "stat", value: "mean")
              |> union(tables: [
                   data |> group(columns: ["location", "metric"]) |> min() |> set(key: "stat", value: "min"),
                   data |> group(columns: ["location", "metric"]) |> max() |> set(key: "stat", value: "max"),
                   data |> group(columns: ["location", "metric"]) |> stddev() |> set(key: "stat", value: "std")
                 ])
              |> pivot(rowKey: ["location", "metric"],
                       columnKey: ["stat"], valueColumn: "_value")
              |> keep(columns: ["location", "metric", "mean", "min", "max", "std"])
        '''

        try:
            tables = self._query_api.query(flux, org=self._org)
            result = {}
            for table in tables:
                for row in table.records:
                    key = f"{row['location']}.{row['metric']}"
                    result[key] = {
                        'mean': round(row.get('mean', 0) or 0, 2),
                        'min': round(row.get('min', 0) or 0, 2),
                        'max': round(row.get('max', 0) or 0, 2),
                        'std': round(row.get('std', 0) or 0, 2)
                    }
            return result
        except Exception as e:
            logger.error(f"统计查询失败: {e}")
            return {}

    # ── 数据管理 ─────────────────────────────────────────────

    def delete_range(self, start: Union[str, datetime],
                     stop: Union[str, datetime],
                     metric: Optional[str] = None,
                     location: Optional[str] = None):
        """
        删除指定时间范围的数据

        Args:
            start:    起始时间
            stop:     结束时间
            metric:   过滤指标
            location: 过滤位置
        """
        self.connect()

        if isinstance(start, datetime):
            start = start.isoformat()
        if isinstance(stop, datetime):
            stop = stop.isoformat()

        predicate = f'_measurement="{self._measurement}"'
        if metric:
            predicate += f' and metric="{metric}"'
        if location:
            predicate += f' and location="{location}"'

        try:
            self._delete_api.delete(
                start=start, stop=stop,
                predicate=predicate,
                bucket=self._bucket, org=self._org
            )
            logger.info(f"已删除数据: {predicate}, 时间 {start} ~ {stop}")
        except Exception as e:
            logger.error(f"删除数据失败: {e}")

    def get_record_count(self, start: str = "-30d",
                         metric: Optional[str] = None,
                         location: Optional[str] = None) -> int:
        """
        获取记录数

        Returns:
            int: 记录条数
        """
        self.connect()

        filters = [f'r._measurement == "{self._measurement}"']
        if metric:
            filters.append(f'r.metric == "{metric}"')
        if location:
            filters.append(f'r.location == "{location}"')
        filter_str = " and ".join(filters)

        flux = f'''
            from(bucket: "{self._bucket}")
              |> range(start: {self._flux_time(start)})
              |> filter(fn: (r) => {filter_str})
              |> filter(fn: (r) => r._field == "value")
              |> count()
        '''

        try:
            tables = self._query_api.query(flux, org=self._org)
            for table in tables:
                for row in table.records:
                    return int(row['_value'])
        except Exception as e:
            logger.error(f"计数查询失败: {e}")
        return 0

    # ── 工具方法 ─────────────────────────────────────────────

    @staticmethod
    def _flux_time(t: Optional[Union[str, datetime]]) -> str:
        """将时间转为 Flux 字符串"""
        if t is None:
            return 'now()'
        if isinstance(t, datetime):
            return t.strftime('%Y-%m-%dT%H:%M:%SZ')
        return t

    @classmethod
    def print_schema(cls):
        """打印 Schema 文档"""
        lines = []
        lines.append("=" * 70)
        lines.append("  InfluxDB 时序数据库 Schema")
        lines.append("=" * 70)
        lines.append("")
        lines.append("  Bucket:      weather_data")
        lines.append("  Measurement: environment_data")
        lines.append("")
        lines.append("  Tags:")
        lines.append("    location   → greenhouse | outdoor")
        lines.append("    metric     → temperature | humidity | temperature_outdoor |")
        lines.append("                  humidity_outdoor | light_intensity |")
        lines.append("                  wind_direction | wind_speed | rainfall")
        lines.append("")
        lines.append("  Fields:")
        lines.append("    value      → float64  测量值")
        lines.append("    quality    → int64    0=正常  1=异常  2=插值")
        lines.append("")
        lines.append("  Timestamp: 对齐到 15 分钟整点")
        lines.append("")
        lines.append("  ┌─────────────────┬────────────┬───────────────────┬───────┐")
        lines.append("  │ location        │ metric     │ unit              │ count │")
        lines.append("  ├─────────────────┼────────────┼───────────────────┼───────┤")
        lines.append("  │ greenhouse      │ temperature│ °C                │    2  │")
        lines.append("  │ greenhouse      │ humidity   │ %                 │      │")
        lines.append("  │ outdoor         │ temp...door│ °C                │    6  │")
        lines.append("  │ outdoor         │ humi...door│ %                 │      │")
        lines.append("  │ outdoor         │ light_inte.│ lux               │      │")
        lines.append("  │ outdoor         │ wind_direc.│ °                 │      │")
        lines.append("  │ outdoor         │ wind_speed │ m/s               │      │")
        lines.append("  │ outdoor         │ rainfall   │ mm                │      │")
        lines.append("  └─────────────────┴────────────┴───────────────────┴───────┘")
        lines.append("")
        lines.append("  预估数据量: 8 metric × 96/day × 365 ≈ 280,320 条/年")
        lines.append("=" * 70)
        return "\n".join(lines)


def main():
    """
    使用示例与 Schema 展示
    """
    print(TimeSeriesDB.print_schema())

    print("\n")
    print("=" * 60)
    print("  使用示例")
    print("=" * 60)

    print("""
# 1. 初始化连接
from tsdb_manager import TimeSeriesDB, round_to_15min

db = TimeSeriesDB(
    url="http://localhost:8086",
    token="your-api-token",
    org="my-org",
    bucket="weather_data"
)

# 2. 初始化 Bucket
db.initialize()

# 3. 单条写入
from datetime import datetime
db.write_point(
    timestamp=datetime.now(),
    location="greenhouse",
    metric="temperature",
    value=23.5,
    quality=0
)

# 4. 批量写入
points = [
    {"time": datetime(2026, 5, 2, 9, 15), "location": "greenhouse",
     "metric": "temperature", "value": 15.4, "quality": 0},
    {"time": datetime(2026, 5, 2, 9, 15), "location": "greenhouse",
     "metric": "humidity", "value": 73.6, "quality": 0},
]
db.write_batch(points)

# 5. 从 DataLoader 导入
from greenhouse_data_loader import GreenhouseDataLoader
from outdoor_weather_loader import OutdoorWeatherLoader

gh_loader = GreenhouseDataLoader("天气数据/温湿度数据/温湿度数据.csv")
gh_loader.load(); gh_loader.check_quality(); gh_loader.standardize()
db.import_from_greenhouse(gh_loader)

ow_loader = OutdoorWeatherLoader("天气数据/宣威市尚营种气象/宣威市尚营种气象.csv")
ow_loader.load(); ow_loader.check_quality(); ow_loader.standardize()
db.import_from_outdoor(ow_loader)

# 6. 时间范围查询
df = db.query_range(start="-1d", metric="temperature", location="greenhouse")
print(df.head())

# 7. 降采样查询（15分钟→小时均值）
df_hourly = db.query_downsample(start="-7d", every="1h")
print(df_hourly.head())

# 8. 查询为宽表
df_wide = db.query_pivot_wide(start="-1d", location="outdoor")
print(df_wide.head())

# 9. 统计查询
stats = db.get_stats(start="-7d")
for k, v in stats.items():
    print(f"  {k}: mean={v['mean']}, min={v['min']}, max={v['max']}")

# 10. 关闭连接
db.close()
""")

    print("\n")
    print("=" * 60)
    print("  时间对齐工具")
    print("=" * 60)

    test_times = [
        datetime(2026, 5, 2, 12, 7, 10),
        datetime(2026, 5, 2, 12, 16, 39),
        datetime(2026, 5, 2, 12, 31, 40),
        datetime(2026, 5, 2, 12, 0, 0),
    ]
    for t in test_times:
        print(f"  {t.strftime('%Y-%m-%d %H:%M:%S')} → {round_to_15min(t).strftime('%Y-%m-%d %H:%M')}")

    print("\n" + "=" * 60)
    print("  本地功能验证（不含数据库连接）通过")
    print("=" * 60)


if __name__ == "__main__":
    main()