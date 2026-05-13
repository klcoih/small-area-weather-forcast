#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气象预测模型训练与导出脚本（集成 Optuna 超参数调优）
自动扫描天气数据文件夹，为每个数据源单独搜索最优超参数、训练模型并导出ONNX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import logging

warnings.filterwarnings('ignore')
# 降低 Optuna 日志级别，避免刷屏
optuna_logger = logging.getLogger('optuna')
optuna_logger.setLevel(logging.WARNING)

# 中文绘图配置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 时序模型依赖库
from darts import TimeSeries
from darts.models import LSTM, AutoformerModel, TimesNetModel
from darts.metrics import mae, rmse, r2_score
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler

# 尝试导入 Optuna 和 PyTorch Lightning
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ 未安装 optuna，将使用默认超参数。安装命令: pip install optuna")

try:
    from pytorch_lightning.callbacks import EarlyStopping
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("⚠️ 未安装 pytorch_lightning，早停功能不可用。安装命令: pip install pytorch-lightning")

# =============================================================================
# 全局配置
# =============================================================================
BASE_DIR = r"D:\pythonProject\昆院智慧大棚小区域气象预测"
DATA_DIR = os.path.join(BASE_DIR, "天气数据")
OUTPUT_DIR = os.path.join(BASE_DIR, "最终结果")

TIME_COL = "时间"

# 时序预测参数
OUTPUT_LEN = 24                  # 预测未来24小时（通常固定）
TRAIN_RATIO = 0.8
FINAL_EPOCHS = 100               # 最终训练的最大轮次（会早停）
FINAL_BATCH_SIZE = 16
TUNING_EPOCHS = 50               # 调优阶段最大轮次（早停更快）
TUNING_BATCH_SIZE = 16

# 超参数搜索配置
ENABLE_TUNING = True             # 是否开启调优（若 optuna 不可用自动关闭）
N_TRIALS = 20                    # 每个模型的搜索试验次数
EARLY_STOP_PATIENCE = 5          # 早停耐心值

# 支持预测的目标列映射
TARGET_MAPPING = {
    '平均温度': 'temperature',
    '温度': 'temperature',
    '平均湿度': 'humidity',
    '湿度': 'humidity',
    '平均光照强度': 'light',
    '光照强度': 'light',
    '平均风向': 'wind_direction',
    '风向': 'wind_direction',
    '平均风速': 'wind_speed',
    '风速': 'wind_speed',
    '平均降雨量': 'rainfall',
    '降雨量': 'rainfall',
    '降水量': 'rainfall',
}


# =============================================================================
# 1. 数据加载与自动检测
# =============================================================================
def load_and_detect_columns(file_path):
    """加载CSV文件并自动检测可用的数据列"""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except:
        try:
            df = pd.read_csv(file_path, encoding="gbk")
        except:
            print(f"  ⚠️ 无法读取文件: {file_path}")
            return None, []

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    df = df.sort_values(TIME_COL).dropna(subset=[TIME_COL])

    available_targets = []
    for col in df.columns:
        if col in TARGET_MAPPING and col != TIME_COL:
            if df[col].notna().sum() > 0:
                available_targets.append(col)

    return df, available_targets


def build_series(df, targets, freq='H'):
    """构建Darts时序数据"""
    try:
        series = TimeSeries.from_dataframe(df, TIME_COL, targets, freq=freq)
        scaler = Scaler(StandardScaler())
        return scaler.fit_transform(series), scaler
    except Exception as e:
        print(f"  ⚠️ 构建时序失败: {e}")
        return None, None


# =============================================================================
# 2. 超参数空间定义
# =============================================================================
def get_lstm_param_space(trial):
    """LSTM 超参数空间"""
    return {
        "input_chunk_length": trial.suggest_categorical("input_chunk_length", [48, 72, 96, 120]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
        "n_rnn_layers": trial.suggest_categorical("n_rnn_layers", [1, 2, 3]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "optimizer_kwargs": {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        }
    }

def get_autoformer_param_space(trial):
    """Autoformer 超参数空间"""
    return {
        "input_chunk_length": trial.suggest_categorical("input_chunk_length", [48, 72, 96, 120]),
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "nhead": trial.suggest_categorical("nhead", [4, 8]),
        "num_encoder_layers": trial.suggest_categorical("num_encoder_layers", [1, 2, 3]),
        "num_decoder_layers": trial.suggest_categorical("num_decoder_layers", [1, 2]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "optimizer_kwargs": {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        }
    }

def get_timesnet_param_space(trial):
    """TimesNet 超参数空间"""
    return {
        "input_chunk_length": trial.suggest_categorical("input_chunk_length", [48, 72, 96, 120]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_categorical("num_layers", [2, 3, 4]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "optimizer_kwargs": {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        }
    }


# =============================================================================
# 3. 调优与训练（集成 Optuna）
# =============================================================================
def create_model_objective(model_class, param_space_fn, train_series, val_series,
                           model_name_prefix, epochs, batch_size):
    """
    创建 Optuna 目标函数
    """
    def objective(trial):
        params = param_space_fn(trial)
        input_len = params.pop("input_chunk_length")

        # 早停回调
        callbacks = []
        if LIGHTNING_AVAILABLE:
            callbacks.append(EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOP_PATIENCE,
                min_delta=0.001
            ))
        pl_trainer_kwargs = {"callbacks": callbacks} if callbacks else {}

        # 构造模型
        model = model_class(
            input_chunk_length=input_len,
            output_chunk_length=OUTPUT_LEN,
            model_name=f"{model_name_prefix}_trial{trial.number}",
            n_epochs=epochs,
            batch_size=batch_size,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            **params
        )

        # 训练
        try:
            model.fit(train_series, val_series=val_series, verbose=False)
        except Exception as e:
            print(f"    ⚠️ 训练失败 (trial {trial.number}): {e}")
            return -float('inf')

        # 评估
        try:
            pred = model.predict(OUTPUT_LEN, series=train_series)
            r2 = r2_score(pred, val_series)
            return r2 if not np.isnan(r2) else -float('inf')
        except Exception:
            return -float('inf')

    return objective


def tune_and_train_models(train_series, val_series, scene_name):
    """
    对三种模型分别进行超参数调优，然后用最佳超参数重新训练最终模型
    """
    models = {}
    best_params_log = {}

    # 模型配置
    model_configs = [
        ("LSTM", LSTM, get_lstm_param_space),
        ("Autoformer", AutoformerModel, get_autoformer_param_space),
        ("TimesNet", TimesNetModel, get_timesnet_param_space),
    ]

    for name, model_cls, param_fn in model_configs:
        print(f"  🔍 调优 {name} ...")
        try:
            if ENABLE_TUNING and OPTUNA_AVAILABLE:
                # 创建研究
                sampler = TPESampler(seed=42)
                study = optuna.create_study(direction="maximize", sampler=sampler)

                objective = create_model_objective(
                    model_cls, param_fn,
                    train_series, val_series,
                    f"{name}_{scene_name}",
                    TUNING_EPOCHS, TUNING_BATCH_SIZE
                )

                # 执行优化
                study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

                best_trial = study.best_trial
                print(f"    ✅ 最佳R²: {best_trial.value:.4f}, 参数: {best_trial.params}")
                best_params_log[name] = best_trial.params

                # 重新训练最终模型
                best_params = param_fn(best_trial)  # 使用最佳参数重新生成（无 trial 对象）
                # 但 best_params 需要从 trial 获取，换成直接使用 study.best_params
                best_params = {k: v for k, v in best_trial.params.items()}
                input_len = best_params.pop("input_chunk_length")
                # 移除 lr 前缀
                lr = best_params.pop("lr")
                final_params = best_params.copy()
                final_params["optimizer_kwargs"] = {"lr": lr}
            else:
                # 无调优，使用默认值
                input_len = 72
                final_params = {}
                if name == "LSTM":
                    final_params = {"hidden_dim": 32, "n_rnn_layers": 2, "dropout": 0.1}
                elif name == "Autoformer":
                    final_params = {"d_model": 64, "nhead": 8, "num_encoder_layers": 2,
                                    "num_decoder_layers": 1, "dropout": 0.1}
                elif name == "TimesNet":
                    final_params = {"hidden_size": 64, "num_layers": 2, "dropout": 0.1}
                default_lr = 0.001
                final_params["optimizer_kwargs"] = {"lr": default_lr}
                best_params_log[name] = {**final_params, "input_chunk_length": input_len}

            # 最终模型训练
            print(f"    🏋️ 最终训练 {name} (input_len={input_len})...")
            callbacks = []
            if LIGHTNING_AVAILABLE:
                callbacks.append(EarlyStopping(
                    monitor="val_loss",
                    patience=EARLY_STOP_PATIENCE,
                    min_delta=0.001
                ))
            pl_trainer_kwargs = {"callbacks": callbacks} if callbacks else {}

            model = model_cls(
                input_chunk_length=input_len,
                output_chunk_length=OUTPUT_LEN,
                model_name=f"{name}_{scene_name}",
                n_epochs=FINAL_EPOCHS,
                batch_size=FINAL_BATCH_SIZE,
                pl_trainer_kwargs=pl_trainer_kwargs,
                force_reset=True,
                **final_params
            )
            model.fit(train_series, val_series=val_series, verbose=False)
            models[name] = model
        except Exception as e:
            print(f"  ⚠️ {name} 调优/训练失败: {e}")
            continue

    return models, best_params_log


# =============================================================================
# 4. 模型评估与选择
# =============================================================================
def evaluate_and_select(models, val_series, scene_name):
    """评估模型性能，选择最优模型"""
    evaluation_results = {}
    best_r2 = -float('inf')
    best_name = None
    best_model = None
    predictions = {}

    for name, model in models.items():
        try:
            pred = model.predict(OUTPUT_LEN, val_series[:model.input_chunk_length])
            r2 = r2_score(pred, val_series)
            mae_val = mae(pred, val_series)
            rmse_val = rmse(pred, val_series)

            evaluation_results[name] = {
                'R2': r2,
                'MAE': mae_val,
                'RMSE': rmse_val,
                'prediction': pred
            }
            predictions[name] = pred

            print(f"  │   {name}: R²={r2:.4f}, MAE={mae_val:.4f}, RMSE={rmse_val:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_name = name
                best_model = model
        except Exception as e:
            print(f"  │   ⚠️ {name}评估失败: {e}")

    return best_model, best_name, best_r2, evaluation_results, predictions


# =============================================================================
# 5. 可视化
# =============================================================================
def create_visualizations(scene_name, val_series, predictions, evaluation_results, best_name, best_params, output_dir):
    """创建可视化图表（增加超参数记录）"""
    try:
        fig = plt.figure(figsize=(16, 14))

        # 1. 模型性能对比柱状图
        ax1 = fig.add_subplot(2, 2, 1)
        model_names = list(evaluation_results.keys())
        r2_scores = [evaluation_results[m]['R2'] for m in model_names]
        colors = ['#2ecc71' if m == best_name else '#3498db' for m in model_names]
        bars = ax1.bar(model_names, r2_scores, color=colors, edgecolor='black')
        ax1.set_ylabel('R² Score', fontsize=12)
        ax1.set_title(f'{scene_name} - 模型R²性能对比', fontsize=14, fontweight='bold')
        ax1.set_ylim(min(r2_scores) - 0.1, 1.0)
        for bar, score in zip(bars, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=10)
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='良好阈值(0.8)')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 2. MAE和RMSE对比
        ax2 = fig.add_subplot(2, 2, 2)
        mae_scores = [evaluation_results[m]['MAE'] for m in model_names]
        rmse_scores = [evaluation_results[m]['RMSE'] for m in model_names]
        x = np.arange(len(model_names))
        width = 0.35
        bars1 = ax2.bar(x - width/2, mae_scores, width, label='MAE', color='#e74c3c')
        bars2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color='#9b59b6')
        ax2.set_ylabel('Error', fontsize=12)
        ax2.set_title(f'{scene_name} - MAE与RMSE对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 3. 预测对比图
        ax3 = fig.add_subplot(2, 1, 2)
        target_names = val_series.columns if hasattr(val_series, 'columns') else ['Value']
        val_values = val_series[-OUTPUT_LEN:].values() if hasattr(val_series[-OUTPUT_LEN:], 'values') else None
        if val_values is not None and len(val_values) > 0:
            time_index = range(len(val_values[0]) if isinstance(val_values, list) else len(val_values))
            ax3.plot(time_index, val_values[0] if isinstance(val_values, list) else val_values,
                    'k-', linewidth=2, label='真实值', alpha=0.8)
            for name, result in evaluation_results.items():
                pred = result['prediction']
                pred_values = pred.values() if hasattr(pred, 'values') else None
                if pred_values is not None and len(pred_values) > 0:
                    linestyle = '-' if name == best_name else '--'
                    linewidth = 2.5 if name == best_name else 1.5
                    ax3.plot(time_index, pred_values[0] if isinstance(pred_values, list) else pred_values,
                            linestyle=linestyle, linewidth=linewidth, label=f'{name}预测', alpha=0.7)
            ax3.set_xlabel('时间步', fontsize=12)
            ax3.set_ylabel(target_names[0], fontsize=12)
            ax3.set_title(f'{scene_name} - 预测结果对比 (未来{OUTPUT_LEN}小时)', fontsize=14, fontweight='bold')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)

        # 添加超参数文本说明
        if best_params:
            param_text = "最优模型超参数:\n" + "\n".join([f"{k}: {v}" for k, v in best_params.items()])
            fig.text(0.02, 0.02, param_text, fontsize=8, family='monospace',
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fig_path = os.path.join(output_dir, f'{scene_name}_可视化评估.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return fig_path
    except Exception as e:
        print(f"  ⚠️ 可视化创建失败: {e}")
        return None


def create_evaluation_chart(evaluation_results, best_name, output_path):
    """创建单独的评估指标图表"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        model_names = list(evaluation_results.keys())

        # R²
        ax1 = axes[0]
        r2_scores = [evaluation_results[m]['R2'] for m in model_names]
        colors = ['#2ecc71' if m == best_name else '#3498db' for m in model_names]
        bars = ax1.bar(model_names, r2_scores, color=colors, edgecolor='black')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² (决定系数)')
        ax1.set_ylim(min(r2_scores) - 0.1, 1.0)
        for bar, score in zip(bars, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=9)
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)

        # MAE
        ax2 = axes[1]
        mae_scores = [evaluation_results[m]['MAE'] for m in model_names]
        colors = ['#2ecc71' if m == best_name else '#e74c3c' for m in model_names]
        bars = ax2.bar(model_names, mae_scores, color=colors, edgecolor='black')
        ax2.set_ylabel('MAE')
        ax2.set_title('MAE (平均绝对误差)')
        for bar, score in zip(bars, mae_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        # RMSE
        ax3 = axes[2]
        rmse_scores = [evaluation_results[m]['RMSE'] for m in model_names]
        colors = ['#2ecc71' if m == best_name else '#9b59b6' for m in model_names]
        bars = ax3.bar(model_names, rmse_scores, color=colors, edgecolor='black')
        ax3.set_ylabel('RMSE')
        ax3.set_title('RMSE (均方根误差)')
        for bar, score in zip(bars, rmse_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path
    except Exception as e:
        print(f"  ⚠️ 评估图表创建失败: {e}")
        return None


# =============================================================================
# 6. 导出ONNX
# =============================================================================
def export_onnx(model, output_path):
    """导出模型为ONNX格式"""
    try:
        model.export_onnx(output_path)
        return True
    except Exception as e:
        print(f"  ⚠️ ONNX导出失败: {e}")
        return False


# =============================================================================
# 7. 保存评估报告
# =============================================================================
def save_evaluation_report(scene_name, evaluation_results, best_name, best_r2,
                           available_targets, best_params, output_path):
    """保存评估报告到CSV（包含超参数）"""
    if not evaluation_results:
        return

    report_data = []
    for model_name, metrics in evaluation_results.items():
        is_best = "★最优" if model_name == best_name else ""
        # 附加最优超参数
        param_str = str(best_params.get(model_name, {})) if best_params else ""
        report_data.append({
            '数据源': scene_name,
            '训练数据列': ', '.join(available_targets),
            '模型': model_name,
            'R2': round(metrics['R2'], 4),
            'MAE': round(metrics['MAE'], 4),
            'RMSE': round(metrics['RMSE'], 4),
            '超参数': param_str,
            '备注': is_best
        })

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_path, index=False, encoding='utf-8')


# =============================================================================
# 主流程
# =============================================================================
def main():
    print("=" * 70)
    print("🚀 智慧大棚气象预测模型训练与导出（Optuna自动调优版）")
    print("=" * 70)

    if not OPTUNA_AVAILABLE:
        print("ℹ️ Optuna未安装，将使用固定超参数。建议安装以获得最佳性能。")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📂 数据目录: {DATA_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")

    subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not subdirs:
        print("❌ 未找到任何数据子文件夹")
        return

    print(f"\n📊 发现 {len(subdirs)} 个数据源: {', '.join(subdirs)}")
    all_results = []

    for subdir in sorted(subdirs):
        print(f"\n{'=' * 70}")
        print(f"📦 处理数据源: {subdir}")
        print(f"{'=' * 70}")
        subdir_path = os.path.join(DATA_DIR, subdir)
        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"⚠️ 数据源 {subdir} 中未找到CSV文件，跳过")
            continue

        source_output_dir = os.path.join(OUTPUT_DIR, subdir)
        os.makedirs(source_output_dir, exist_ok=True)

        for csv_file in sorted(csv_files):
            csv_path = os.path.join(subdir_path, csv_file)
            file_name = os.path.splitext(csv_file)[0]

            print(f"\n{'─' * 50}")
            print(f"📄 文件: {csv_file}")
            print(f"{'─' * 50}")

            df, available_targets = load_and_detect_columns(csv_path)
            if df is None or df.empty:
                print(f"⚠️ 无法加载数据，跳过")
                continue

            print(f"  📊 数据条数: {len(df)}")
            print(f"  🎯 可用数据列: {', '.join(available_targets) if available_targets else '无'}")

            if not available_targets:
                print(f"⚠️ 没有可用的预测目标列，跳过")
                continue

            series, scaler = build_series(df, available_targets)
            if series is None:
                print(f"⚠️ 无法构建时序数据，跳过")
                continue

            train_series, val_series = series.split_after(TRAIN_RATIO)
            if len(train_series) < 72:  # 至少需要满足最小的 input_chunk_length
                print(f"⚠️ 训练数据不足({len(train_series)} < 72)，跳过")
                continue

            print(f"  ✅ 训练集: {len(train_series)} 条, 验证集: {len(val_series)} 条")

            # 调优与训练
            print(f"\n  🤖 超参数调优与模型训练...")
            models, best_params_log = tune_and_train_models(train_series, val_series, file_name)

            if not models:
                print(f"⚠️ 没有模型训练成功，跳过")
                continue

            # 评估
            print(f"\n  📈 模型评估结果:")
            best_model, best_name, best_r2, eval_results, predictions = evaluate_and_select(
                models, val_series, file_name
            )
            if best_model is None:
                print(f"⚠️ 模型评估失败，跳过")
                continue

            print(f"\n  🎉 最优模型: {best_name} (R²={best_r2:.4f})")
            best_params = best_params_log.get(best_name, {})
            print(f"  🧪 最优超参数: {best_params}")

            # 可视化
            print(f"\n  📊 生成可视化图表...")
            fig_path = create_visualizations(file_name, val_series, predictions, eval_results, best_name,
                                             best_params, source_output_dir)
            if fig_path:
                print(f"  ✅ 综合图表已保存: {fig_path}")

            eval_chart_path = os.path.join(source_output_dir, f"{file_name}_评估指标.png")
            create_evaluation_chart(eval_results, best_name, eval_chart_path)
            print(f"  ✅ 评估指标图表已保存: {eval_chart_path}")

            # 导出ONNX
            onnx_path = os.path.join(source_output_dir, f"{file_name}_最优模型.onnx")
            if export_onnx(best_model, onnx_path):
                print(f"  ✅ ONNX已导出: {onnx_path}")

            # 保存评估报告
            report_path = os.path.join(source_output_dir, f"{file_name}_评估报告.csv")
            save_evaluation_report(file_name, eval_results, best_name, best_r2,
                                   available_targets, best_params_log, report_path)
            print(f"  ✅ 评估报告已保存: {report_path}")

            all_results.append({
                '数据源': subdir,
                '文件名': csv_file,
                '数据条数': len(df),
                '可用列': ', '.join(available_targets),
                '最优模型': best_name,
                'R2': round(best_r2, 4)
            })

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(OUTPUT_DIR, "模型训练总报告.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        print(f"\n{'=' * 70}")
        print(f"📋 模型训练总报告已保存: {summary_path}")
        create_summary_chart(all_results, OUTPUT_DIR)

    print(f"\n{'=' * 70}")
    print("🏆 所有数据源模型训练完成!")
    print(f"{'=' * 70}")


def create_summary_chart(all_results, output_dir):
    """创建所有数据源的整体对比图表"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        df_summary = pd.DataFrame(all_results)
        x = np.arange(len(df_summary))
        colors = plt.cm.Set3(np.linspace(0, 1, len(df_summary)))
        bars = ax.bar(x, df_summary['R2'], color=colors, edgecolor='black')
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('各数据源最优模型R²对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_summary['数据源'] + '\n' + df_summary['最优模型'], rotation=0, ha='center')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='良好阈值(0.8)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        for bar, r2, model in zip(bars, df_summary['R2'], df_summary['最优模型']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{r2:.4f}\n({model})', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        summary_fig_path = os.path.join(output_dir, "模型总览对比.png")
        plt.savefig(summary_fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✅ 总览图表已保存: {summary_fig_path}")
    except Exception as e:
        print(f"  ⚠️ 总览图表创建失败: {e}")


if __name__ == "__main__":
    main()