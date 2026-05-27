#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 模型（PyTorch nn.LSTM）

适用目标: 所有目标（复杂非线性）
可调参数:
  - num_layers: LSTM 层数 (1-4)
  - hidden_size: 隐藏单元数 (32-256)
  - dropout: 丢弃率 (0.0-0.5)
  - learning_rate: 学习率 (1e-4 ~ 1e-2)
  - seq_length: 序列长度 (16-192, 对应4-48小时)
  - batch_size: 批量大小 (16-128)
  - epochs: 训练轮数 (10-100)

特殊:
  - 风向: 输出层2神经元 (sin, cos)
  - 降雨: 双输出头 (分类 + 回归)
"""

import time
import logging
import numpy as np
import copy

logger = logging.getLogger(__name__)


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class LSTMModel:
    """
    通用 LSTM 回归模型

    Args:
        input_size:  输入特征维度
        hidden_size: 隐藏单元数
        num_layers:  LSTM 层数
        dropout:     Dropout 率
        output_size: 输出维度 (1=回归, 2=sin/cos)
        learning_rate: 学习率
        device:      'cpu' / 'cuda'
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2,
                 output_size=1, learning_rate=0.001, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None

    def _build_model(self):
        import torch
        import torch.nn as nn

        class LSTMPredictor(nn.Module):
            def __init__(self, input_sz, hidden_sz, num_layers, dropout, output_sz):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_sz, hidden_sz, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0
                )
                self.fc = nn.Linear(hidden_sz, output_sz)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self.model = LSTMPredictor(
            self.input_size, self.hidden_size,
            self.num_layers, self.dropout, self.output_size
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            seq_length=96, batch_size=64, epochs=30, patience=10):
        """
        训练 LSTM

        Args:
            X_train: (n_samples, n_features) 或 (n_samples, seq_length, n_features)
            y_train: (n_samples, output_size)
            X_val, y_val: 验证数据
            seq_length: 序列长度
            batch_size: 批量大小
            epochs: 训练轮数
            patience: 早停耐心值
        """
        import torch

        self._build_model()

        if X_train.ndim == 2:
            X_train, y_train = _to_sequences(
                X_train, y_train, seq_length, self.output_size
            )
        if X_val is not None and X_val.ndim == 2:
            X_val, y_val = _to_sequences(
                X_val, y_val, seq_length, self.output_size
            )

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        early_stopper = EarlyStopping(patience=patience)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(batch_x)
            train_loss /= len(dataset)

            val_loss = train_loss
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    y_v = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    pred_v = self.model(X_v)
                    val_loss = self.criterion(pred_v, y_v).item()

            if early_stopper(val_loss, self.model):
                logger.info(f"LSTM 早停于 epoch {epoch + 1}")
                break

        early_stopper.restore(self.model)
        logger.info(
            f"LSTM 训练完成: hidden={self.hidden_size}, layers={self.num_layers}, "
            f"seq_len={seq_length}"
        )

    def predict(self, X):
        """预测"""
        import torch

        self.model.eval()
        if len(X.shape) == 2:
            n = len(X)
            batch = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            batch = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out = self.model(batch).cpu().numpy()
        return out


def _to_sequences(X, y, seq_length, output_size=1):
    """将 (n_samples, n_features) 转为 (n_samples-seq_length, seq_length, n_features)"""
    if isinstance(y, np.ndarray) and y.ndim == 1:
        y = y.reshape(-1, 1)

    X_seq = np.zeros((len(X) - seq_length, seq_length, X.shape[1]), dtype=np.float32)
    y_out = np.zeros((len(X) - seq_length, output_size), dtype=np.float32)

    for i in range(len(X) - seq_length):
        X_seq[i] = X[i:i + seq_length]
        if y.ndim == 1:
            y_out[i, 0] = y[i + seq_length]
        else:
            y_out[i] = y[i + seq_length]

    return X_seq, y_out


def train_lstm_regression(X_train, y_train, X_val, y_val, params=None):
    """
    训练 LSTM 回归模型

    Returns:
        (y_pred, metrics, train_time)
    """
    from utils.metrics import mae, rmse, r2_score

    if params is None:
        params = {}

    input_size = X_train.shape[1] if X_train.ndim == 2 else X_train.shape[2]
    seq_length = params.get('seq_length', 96)
    batch_size = params.get('batch_size', 64)
    epochs = params.get('epochs', 30)

    t0 = time.time()
    model = LSTMModel(
        input_size=input_size,
        hidden_size=params.get('hidden_size', 64),
        num_layers=params.get('num_layers', 2),
        dropout=params.get('dropout', 0.2),
        output_size=1,
        learning_rate=params.get('learning_rate', 0.001),
        device=params.get('device', 'cpu'),
    )
    model.fit(X_train, y_train, X_val, y_val,
              seq_length=seq_length, batch_size=batch_size, epochs=epochs)
    y_pred = model.predict(X_val)
    train_time = time.time() - t0

    y_pred = y_pred.flatten()
    y_val = np.array(y_val, dtype=float).flatten()
    min_len = min(len(y_pred), len(y_val))
    y_pred = y_pred[:min_len]
    y_val = y_val[:min_len]

    return y_pred, {
        'mae': mae(y_val, y_pred),
        'rmse': rmse(y_val, y_pred),
        'r2': r2_score(y_val, y_pred),
    }, train_time


def train_lstm_wind(X_train, y_train_sin, y_train_cos,
                    X_val, y_val_sin, y_val_cos, params=None):
    """
    LSTM 风向预测 (输出层2神经元: sin, cos)

    Returns:
        (np.column_stack([sin_pred, cos_pred]), metrics, train_time)
    """
    from utils.metrics import angular_error_from_components

    if params is None:
        params = {}

    input_size = X_train.shape[1] if X_train.ndim == 2 else X_train.shape[2]
    seq_length = params.get('seq_length', 96)

    y_train = np.column_stack([np.array(y_train_sin), np.array(y_train_cos)])

    t0 = time.time()
    model = LSTMModel(
        input_size=input_size,
        hidden_size=params.get('hidden_size', 64),
        num_layers=params.get('num_layers', 2),
        dropout=params.get('dropout', 0.2),
        output_size=2,
        learning_rate=params.get('learning_rate', 0.001),
        device=params.get('device', 'cpu'),
    )
    model.fit(X_train, y_train, X_val, np.column_stack([np.array(y_val_sin), np.array(y_val_cos)]),
              seq_length=seq_length, batch_size=params.get('batch_size', 64),
              epochs=params.get('epochs', 30))
    y_pred = model.predict(X_val)
    train_time = time.time() - t0

    sin_pred = y_pred[:, 0].flatten()
    cos_pred = y_pred[:, 1].flatten()
    y_sin = np.array(y_val_sin).flatten()
    y_cos = np.array(y_val_cos).flatten()
    min_len = min(len(sin_pred), len(y_sin))
    angular_err = angular_error_from_components(
        y_sin[:min_len], y_cos[:min_len], sin_pred[:min_len], cos_pred[:min_len]
    )

    return np.column_stack([sin_pred, cos_pred]), {
        'angular_error': angular_err,
    }, train_time


def train_lstm_rainfall_two_stage(X_train, y_train_flag, y_train_rain,
                                  X_val, y_val_flag, y_val_rain, params=None):
    """
    LSTM 降雨两阶段预测:
      分类头: sigmoid -> rain_flag probability
      回归头: linear -> rainfall amount

    Returns:
        ((y_prob, y_pred_rain), metrics, train_time)
    """
    from utils.metrics import auc, brier_score, csi, mae

    if params is None:
        params = {}

    input_size = X_train.shape[1] if X_train.ndim == 2 else X_train.shape[2]
    seq_length = params.get('seq_length', 96)

    y_train_combined = np.column_stack([
        np.array(y_train_flag, dtype=float),
        np.array(y_train_rain, dtype=float)
    ])

    t0 = time.time()
    model = LSTMModel(
        input_size=input_size,
        hidden_size=params.get('hidden_size', 64),
        num_layers=params.get('num_layers', 2),
        dropout=params.get('dropout', 0.2),
        output_size=2,
        learning_rate=params.get('learning_rate', 0.001),
        device=params.get('device', 'cpu'),
    )

    y_val_combined = np.column_stack([
        np.array(y_val_flag, dtype=float),
        np.array(y_val_rain, dtype=float)
    ])

    model.fit(X_train, y_train_combined, X_val, y_val_combined,
              seq_length=seq_length, batch_size=params.get('batch_size', 64),
              epochs=params.get('epochs', 30))

    y_pred = model.predict(X_val)
    train_time = time.time() - t0

    y_prob = y_pred[:, 0]
    y_pred_rain = y_pred[:, 1]
    y_prob = np.clip(y_prob, 0, 1)
    y_pred_binary = (y_prob >= 0.5).astype(int)

    y_val_f = np.array(y_val_flag).flatten()
    y_val_r = np.array(y_val_rain).flatten()
    min_len = min(len(y_prob), len(y_val_f))
    y_prob = y_prob[:min_len]
    y_pred_rain = y_pred_rain[:min_len]
    y_val_f = y_val_f[:min_len]
    y_val_r = y_val_r[:min_len]

    cls_metrics = {
        'auc': auc(y_val_f, y_prob),
        'brier': brier_score(y_val_f, y_prob),
        'csi': csi(y_val_f, y_pred_binary),
    }

    rain_mask = y_val_f == 1
    rain_mae = mae(y_val_r[rain_mask], y_pred_rain[rain_mask]) if rain_mask.sum() > 0 else 0.0

    return (y_prob, y_pred_rain), {
        'classification': cls_metrics,
        'regression': {'mae_rain_only': rain_mae},
    }, train_time