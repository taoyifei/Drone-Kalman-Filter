"""Kalman 状态传播模型工具。"""

from __future__ import annotations

import numpy as np

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import ParsedMessage


def effective_dt(previous: ParsedMessage, current: ParsedMessage,
                 config: PluginConfig) -> float:
    """计算用于状态传播的有效时间步长。

    Args:
        previous: 前一个观测或时间点。
        current: 当前观测或时间点。
        config: 插件配置。

    Returns:
        float: 应用于滤波计算的有效时间间隔。
    """
    delta = (current.event_time - previous.event_time).total_seconds()
    return max(delta, config.min_dt_seconds)


def transition_matrix(dt: float) -> np.ndarray:
    """生成二维恒速模型的状态转移矩阵。

    Args:
        dt: 时间间隔，单位为秒。

    Returns:
        np.ndarray: 对应时间间隔的状态转移矩阵。
    """
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def process_noise(dt: float, sigma_acceleration: float) -> np.ndarray:
    """生成白噪声加速度假设下的过程噪声矩阵。

    Args:
        dt: 时间间隔，单位为秒。
        sigma_acceleration: 过程加速度标准差。

    Returns:
        np.ndarray: 对应时间间隔的过程噪声协方差矩阵。
    """
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    acceleration_variance = sigma_acceleration**2
    return acceleration_variance * np.array(
        [
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ],
        dtype=float,
    )
