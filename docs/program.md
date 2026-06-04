# =============================================
# program.md - 大棚与户外环境预测
# =============================================

## Baseline 指标（根据实际数据设定）

| 指标 | 基线值 | 说明 |
|------|--------|------|
| `val_mae_greenhouse_temp` | 1.0 | 大棚温度 MAE (°C) |
| `val_mae_greenhouse_humidity` | 3.0 | 大棚湿度 MAE (%) |
| `val_mae_outdoor_temp` | 1.2 | 户外温度 MAE (°C) |
| `val_mae_outdoor_humidity` | 4.0 | 户外湿度 MAE (%) |
| `val_mae_light_intensity` | 5000 | 光照强度 MAE (lux) |
| `val_mae_wind_speed` | 0.5 | 风速 MAE (m/s) |
| `val_wind_direction_error` | 30 | 风向角度误差 (°) |
| `val_rain_csi` | 0.6 | 降雨临界成功指数 CSI |

**资源约束：**

| 约束项 | 值 |
|--------|-----|
| `peak_vram` | 8 GB |
| `train_time_limit` | 600 秒 |

---

# =============================================
# 预测目标与推荐模型
# =============================================

## 目标1-2：大棚温湿度

- **文件内名称:** `greenhouse_temperature`, `greenhouse_humidity`
- **特点:** 15分钟间隔，受大棚结构保温影响，波动较户外缓和
- **推荐模型:** LSTM、XGBoost
- **关键特征:** 滞后1-4（15分钟-1小时）、日周期 `lag_96`、户外温度、户外湿度
- **评估指标:** MAE, RMSE, R² → `val_composite`
- **候选模型（按优先级）:**
  1. XGBoost — 特征丰富场景首选，快速可靠
  2. LSTM — 复杂非线性模式，seq_length=96 捕获日周期
  3. ARIMA — 单变量基线，季节性 m=96
  4. Prophet — 周期性建模基线

---

## 目标3-4：户外温湿度

- **文件内名称:** `outdoor_temperature`, `outdoor_humidity`
- **特点:** 15分钟间隔，受气象影响大，有明显的日周期和年周期
- **推荐模型:** Prophet、XGBoost、ARIMA
- **关键特征:** 日周期 `lag_96`、季节特征(month, day_of_week)、风速、光照
- **评估指标:** MAE, RMSE, R² → `val_composite`
- **候选模型（按优先级）:**
  1. Prophet — 天然适合周期性气象数据，daily_seasonality=True
  2. XGBoost — 多特征融合，滞后特征丰富
  3. ARIMA — 季节性 ARIMA，单变量快速基线
  4. LSTM — 序列模型，需要更多调参

---

## 目标5：光照强度

- **文件内名称:** `light_intensity`
- **特点:** 强日周期，夜间为0，白天波动大
- **推荐模型:** Prophet（首选）、LSTM
- **特殊处理:** 夜间标记 `is_night` 特征，日周期特征
- **评估指标:** MAE, RMSE, R² → `val_composite`
- **候选模型（按优先级）:**
  1. Prophet — 强周期性数据，daily_seasonality + yearly_seasonality
  2. XGBoost — 使用 `is_daytime` 和 `hour_sin/cos` 编码
  3. LSTM — seq_length=96 捕获日模式

---

## 目标6：风向

- **文件内名称:** `wind_direction`
- **特点:** 环形数据 (0-360°)，静风时风向无实际意义
- **推荐模型:** LSTM（首选）、XGBoost
- **特殊处理:**
  - **角度→向量转换:** `(wind_sin, wind_cos) = (sin(θ), cos(θ))`，输出层2神经元
  - **静风处理:** 风速 < 0.5 m/s 时，风向样本权重降低或排除
  - **损失函数:** 向量 MSE = (sin_true - sin_pred)² + (cos_true - cos_pred)²
  - **评估指标:** 平均绝对角度误差 `angular_error` → `val_composite`
- **候选模型（按优先级）:**
  1. LSTM — 双输出头 (sin, cos)，直接优化向量损失
  2. XGBoost — 训练两个回归器分别预测 sin 和 cos

---

## 目标7：风速

- **文件内名称:** `wind_speed`
- **特点:** 右偏分布，间歇性强，大风事件稀少但重要
- **推荐模型:** XGBoost（首选）、LSTM
- **特殊处理:**
  - **加权 MSE:** 样本权重 = 1 + wind_speed（大风样本更高权重）
  - 捕获大风事件比平均精度更重要
- **评估指标:** MAE, RMSE, R² → `val_composite`
- **候选模型（按优先级）:**
  1. XGBoost — 加权回归 `sample_weight=1+wind_speed`
  2. LSTM — 自定义加权 MSE 损失

---

## 目标8：降雨量

- **文件内名称:** `rainfall`
- **特点:** 零膨胀（大量0值），非零部分为右偏分布
- **推荐模型:** XGBoost 两阶段（首选）、马尔可夫链
- **两阶段流程:**
  - **阶段1 - 分类:** 预测有雨/无雨 (`rain_flag`, 阈值 0.1mm)
    - 模型: XGBClassifier
    - 指标: AUC, Brier Score, CSI
  - **阶段2 - 回归:** 仅降雨样本预测降雨量
    - 模型: XGBRegressor
    - 变换: `log1p(rainfall)` 减轻右偏
    - 指标: MAE (rain_only)
  - **合成指标:** `val_composite` = 0.5×cls_score + 0.5×reg_score
- **候选模型（按优先级）:**
  1. XGBoost 两阶段 — 最可靠的两阶段实现
  2. LSTM 双输出头 — 端到端分类+回归
  3. Markov Chain — 状态转移概率，基线对比

---

# =============================================
# 各模型调优方向
# =============================================

## ARIMA 调优

- 尝试不同的 `max_p`, `max_d`, `max_q`（0-5）
- 尝试 `seasonal=True` / `seasonal=False`
- 尝试 `m=96`（日周期，15分钟×96=1天）
- 适用目标：温度、湿度

**调参网格:**
```yaml
arima:
  seasonal: [true, false]
  m: [96]
  max_p: [3, 5]
  max_d: [1, 2]
  max_q: [3, 5]
```

---

## Prophet 调优

- `changepoint_prior_scale`: [0.001, 0.01, 0.05, 0.1, 0.5]
- `seasonality_prior_scale`: [0.01, 0.1, 1.0, 10.0]
- `daily_seasonality`: True（必须，15分钟数据）
- `yearly_seasonality`: True / False
- 适用目标：温度、湿度、光照强度

**调参网格:**
```yaml
prophet:
  changepoint_prior_scale: [0.01, 0.05, 0.1, 0.5]
  seasonality_prior_scale: [1.0, 5.0, 10.0]
  yearly_seasonality: [true, false]
  daily_seasonality: [true]
```

---

## XGBoost 调优

- `max_depth`: [3, 5, 7, 10]
- `learning_rate`: [0.01, 0.05, 0.1, 0.2]
- `n_estimators`: [100, 200, 500]
- `subsample`: [0.6, 0.8, 1.0]
- `colsample_bytree`: [0.6, 0.8, 1.0]
- `reg_alpha`: [0, 0.1, 1.0]
- `reg_lambda`: [0.1, 1.0, 10.0]
- 特征选择：尝试不同的滞后窗口组合
- 风速：尝试加权损失函数 (`sample_weight=1+wind_speed`)
- 降雨：尝试不同的分类阈值 (`threshold=0.05, 0.1, 0.2`)

**调参网格:**
```yaml
xgboost:
  max_depth: [5, 7, 10]
  learning_rate: [0.01, 0.05, 0.1]
  n_estimators: [200, 500, 800]
  subsample: [0.8, 1.0]
  colsample_bytree: [0.8, 1.0]
  reg_alpha: [0, 0.1]
  reg_lambda: [1.0]
```

---

## LSTM 调优

- `num_layers`: [1, 2, 3]
- `hidden_size`: [32, 64, 128, 256]
- `dropout`: [0.0, 0.1, 0.2, 0.3]
- `learning_rate`: [1e-4, 5e-4, 1e-3, 5e-3]
- `seq_length`: [16, 32, 64, 96, 192]（4小时-48小时）
- `batch_size`: [16, 32, 64]
- `epochs`: [20, 30, 50]
- `optimizer`: Adam（默认）
- 风向：尝试不同的输出头设计（双头sin/cos vs 直接角度回归）
- 降雨：尝试双输出头 vs 两阶段

**调参网格:**
```yaml
lstm:
  num_layers: [2, 3]
  hidden_size: [64, 128]
  dropout: [0.2, 0.3]
  learning_rate: [0.001]
  seq_length: [64, 96, 192]
  batch_size: [32, 64]
  epochs: [30, 50]
```

---

## 马尔可夫链调优

- `n_states`: 降雨 [2, 3, 4]，风向 [8, 16]
- `thresholds`: 尝试不同的分类阈值
- `n_steps`: [1, 3, 6, 12]（15分钟-3小时）
- 适用目标：降雨 (rain/no-rain 状态转移)

**调参网格:**
```yaml
markov:
  n_states: [2, 4]
  thresholds: [[0.1], [0.1, 1.0, 5.0]]
  n_steps: [1]
```

---

# =============================================
# 特征工程优化
# =============================================

## 滞后特征窗口

| 类别 | 滞后步数 | 时间跨度 | 用途 |
|------|----------|----------|------|
| 短期 | `lag_1` | 15分钟 | 最近变化趋势 |
| 短期 | `lag_2` | 30分钟 | 半小时变化 |
| 短期 | `lag_4` | 1小时 | 小时变化 |
| 中期 | `lag_16` | 4小时 | 半日趋势 |
| 中期 | `lag_32` | 8小时 | 半日周期 |
| 中期 | `lag_48` | 12小时 | 半日周期 |
| 长期 | `lag_96` | 1天 | 日周期（核心） |
| 长期 | `lag_192` | 2天 | 2天前同期 |
| 长期 | `lag_672` | 1周 | 周周期 |

## 时间特征

| 特征 | 类型 | 说明 |
|------|------|------|
| `hour` | int (0-23) | 小时 |
| `minute` | int (0/15/30/45) | 分钟 |
| `day_of_week` | int (0-6) | 星期 |
| `month` | int (1-12) | 月份 |
| `is_daytime` | binary | 根据光照强度或时间判断 |
| `hour_sin` | float [-1, 1] | sin(2π × hour / 24) |
| `hour_cos` | float [-1, 1] | cos(2π × hour / 24) |

## 统计特征

| 特征 | 窗口 | 说明 |
|------|------|------|
| `rolling_mean_4` | 1小时 (4×15min) | 短期均值 |
| `rolling_mean_16` | 4小时 (16×15min) | 中期均值 |
| `rolling_std_4` | 1小时 (4×15min) | 短期波动 |

## 交叉特征（跨目标关联）

| 特征 | 公式 | 说明 |
|------|------|------|
| `temp_diff` | temp_greenhouse - temp_outdoor | 大棚保温效应 |
| `humidity_diff` | humidity_greenhouse - humidity_outdoor | 大棚保湿效应 |
| `temp_humidity_interaction` | temperature × humidity | 温湿度交互 |

---

# =============================================
# 实验执行顺序（推荐优先级）
# =============================================

## 第一轮：快速基线

| # | 模型 | 目标 | 预计耗时 | 目的 |
|---|------|------|----------|------|
| 1 | XGBoost | greenhouse_temperature | ~5s | XGBoost 温度基线 |
| 2 | XGBoost | greenhouse_humidity | ~5s | XGBoost 湿度基线 |
| 3 | XGBoost | outdoor_temperature | ~10s | XGBoost 户外温度基线 |
| 4 | ARIMA | greenhouse_temperature | ~30s | 单变量温度基线 |
| 5 | ARIMA | outdoor_temperature | ~30s | 单变量户外温度基线 |
| 6 | Prophet | outdoor_temperature | ~20s | 周期性户外温度基线 |

## 第二轮：多模型交叉

| # | 模型 | 目标 | 预计耗时 | 目的 |
|---|------|------|----------|------|
| 7 | LSTM | greenhouse_temperature | ~90s | 深度学习温度 |
| 8 | XGBoost | light_intensity | ~10s | 光照基线 |
| 9 | Prophet | light_intensity | ~20s | 周期性光照 |
| 10 | XGBoost | wind_speed | ~10s | 加权风速 |
| 11 | XGBoost | wind_direction | ~10s | sin/cos 风向 |
| 12 | LSTM | wind_direction | ~90s | LSTM 风向 |

## 第三轮：特殊目标

| # | 模型 | 目标 | 预计耗时 | 目的 |
|---|------|------|----------|------|
| 13 | XGBoost | rainfall | ~10s | 两阶段降雨 |
| 14 | Markov | rainfall | ~5s | 状态转移降雨基线 |
| 15 | LSTM | rainfall | ~90s | LSTM 双头降雨 |

## 第四轮：调优

| # | 模型 | 目标 | 预计耗时 | 目的 |
|---|------|------|----------|------|
| 16 | XGBoost (调参) | 所有目标 | ~30s/目标 | 网格搜索最优 XGBoost |
| 17 | LSTM (调参) | 温度类目标 | ~2min/目标 | 网格搜索最优 LSTM |

---

# =============================================
# 约束
# =============================================

- 训练时间不超过 600 秒（10 分钟）
- GPU 显存不超过 8 GB
- 保持代码简洁，不要过度复杂化
- **简单优于复杂：** 一个小改进如果增加大量复杂度，不值得保留
- **不要修改 prepare.py**

---

# =============================================
# 关键指令
# =============================================

**NEVER STOP.** 一旦开始实验循环，不要停下来问人类是否继续。
除非是重大崩溃（如 OOM、数据加载失败），否则直接运行下一个实验。
修复小错误（如 typo、import 错误）后继续，不要停下来。

## 实验循环伪代码

```python
for model in [xgboost, lstm, arima, prophet, markov]:
    for target in available_targets:
        if model not in target.supported_models:
            continue
        
        # 1. 从 config.yaml 读取超参网格
        # 2. 遍历参数组合
        # 3. 训练 + 验证评估
        # 4. 写入 results.csv
        # 5. 立即开始下一个实验（不问人类）
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `results.csv` | 所有实验结果汇总，train.py 运行后自动追加 |
| `config.yaml` | 超参数网格配置，AutoResearch 搜索空间 |
| `autoresearch_data/` | prepare.py 生成的训练/验证数据 |