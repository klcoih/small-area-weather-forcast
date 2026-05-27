#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
马尔可夫链模型（numpy/scipy 实现转移概率矩阵）

适用目标: 降雨（有雨/无雨）、风向（离散方向）

可调参数:
  - n_states: 状态数 (降雨: 2-4类, 风向: 8-16方向)
  - thresholds: 状态分类阈值列表
  - n_steps: 预测步数

实现: Chapman-Kolmogorov 方程多步预测 (P^n)
"""

import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MarkovChain:
    """
    一阶马尔可夫链

    使用转移概率矩阵 P (n_states × n_states):
      P[i, j] = P(next=j | current=i)
    多步预测: P_k = P^k (Chapman-Kolmogorov 方程)
    """

    def __init__(self, n_states=2, thresholds=None, n_steps=1):
        self.n_states = n_states
        self.thresholds = thresholds or [0.1]
        self.n_steps = n_steps
        self.transition_matrix = None
        self.stationary_dist = None
        self.state_labels = None

    def _discretize(self, values):
        """将连续值离散化为状态 0, 1, 2, ..."""
        v = np.array(values, dtype=float)
        states = np.zeros(len(v), dtype=int)
        boundaries = sorted(self.thresholds)
        for i, b in enumerate(boundaries):
            states[v >= b] = i + 1
        return states

    def fit(self, y_train):
        """
        训练：构建转移概率矩阵

        Args:
            y_train: 1D 连续值或状态序列
        """
        y = np.array(y_train, dtype=float)

        if self.thresholds and np.any(y != y.astype(int)):
            states = self._discretize(y)
        else:
            states = y.astype(int)

        unique_states = np.unique(states)
        actual_n = len(unique_states)
        self.n_states = max(self.n_states, actual_n)

        self.transition_matrix = np.zeros((self.n_states, self.n_states))

        for i in range(len(states) - 1):
            s_from = states[i]
            s_to = states[i + 1]
            if s_from < self.n_states and s_to < self.n_states:
                self.transition_matrix[s_from, s_to] += 1

        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix /= row_sums

        self._compute_stationary()

        self.state_labels = {i: f"state_{i}" for i in range(self.n_states)}
        logger.info(
            f"Markov: n_states={self.n_states}, "
            f"transition matrix shape={self.transition_matrix.shape}"
        )

    def _compute_stationary(self):
        """计算稳态分布 (eigenvector for eigenvalue 1)"""
        try:
            eigvals, eigvecs = np.linalg.eig(self.transition_matrix.T)
            idx = np.argmin(np.abs(eigvals - 1.0))
            self.stationary_dist = np.real(eigvecs[:, idx])
            self.stationary_dist = np.abs(self.stationary_dist)
            self.stationary_dist /= self.stationary_dist.sum()
        except Exception:
            self.stationary_dist = np.ones(self.n_states) / self.n_states

    def predict(self, n_steps=None, initial_state=None):
        """
        预测

        Args:
            n_steps: 预测步数 (None 则使用 self.n_steps)
            initial_state: 初始状态 (None 则使用稳态分布)

        Returns:
            state_probs: (n_steps, n_states) 每步的状态概率分布
            states: (n_steps,) 最可能的状态序列
        """
        if self.transition_matrix is None:
            raise RuntimeError("模型未训练, 请先调用 fit()")

        steps = n_steps or self.n_steps

        if initial_state is None:
            state_dist = self.stationary_dist.copy()
        elif isinstance(initial_state, (int, np.integer)):
            state_dist = np.zeros(self.n_states)
            state_dist[min(initial_state, self.n_states - 1)] = 1.0
        else:
            state_dist = np.array(initial_state, dtype=float)
            state_dist /= state_dist.sum()

        probs = [state_dist.copy()]
        for _ in range(steps):
            state_dist = state_dist @ self.transition_matrix
            probs.append(state_dist.copy())

        probs = np.array(probs[1:])
        states = np.argmax(probs, axis=1)
        return probs, states

    def predict_next(self, current_state):
        """单步预测"""
        _, states = self.predict(n_steps=1, initial_state=current_state)
        return states[0] if len(states) > 0 else current_state


def train_markov(y_train, y_val, params=None):
    """
    训练马尔可夫链并评估

    Args:
        y_train: 训练序列
        y_val:   验证序列
        params:  参数字典

    Returns:
        (y_pred_states, metrics, train_time)
    """
    from utils.metrics import csi

    if params is None:
        params = {}

    t0 = time.time()
    model = MarkovChain(**params)
    model.fit(y_train)

    yt = np.array(y_train, dtype=float)
    yv = np.array(y_val, dtype=float)

    if model.thresholds and np.any(yt != yt.astype(int)):
        val_states = model._discretize(yv)
        val_states_train = model._discretize(yt)
    else:
        val_states = yv.astype(int)
        val_states_train = yt.astype(int)

    pred_states = []
    if len(val_states_train) > 0:
        current = val_states_train[-1]
        for _ in range(len(val_states)):
            current = model.predict_next(current)
            pred_states.append(current)
    else:
        pred_states = np.zeros(len(val_states), dtype=int)

    pred_states = np.array(pred_states)
    train_time = time.time() - t0

    if model.n_states == 2:
        csi_val = csi(val_states, pred_states)
    else:
        csi_val = csi((val_states >= 1).astype(int), (pred_states >= 1).astype(int))

    return pred_states, {
        'csi': csi_val,
        'accuracy': np.mean(val_states == pred_states),
    }, train_time