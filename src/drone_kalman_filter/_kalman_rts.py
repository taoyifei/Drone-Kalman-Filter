"""前向 Kalman 与后向 RTS 的内部实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from drone_kalman_filter._kalman_model import (
    effective_dt,
    process_noise,
    transition_matrix,
)
from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import ParsedMessage


@dataclass(frozen=True, slots=True)
class FilterPass:
    """保存前向滤波阶段的全部中间结果。"""

    filtered_states: list[np.ndarray]
    filtered_covariances: list[np.ndarray]
    predicted_states: list[np.ndarray | None]
    predicted_covariances: list[np.ndarray | None]
    transition_matrices: list[np.ndarray | None]


def forward_filter(
    observations: Sequence[ParsedMessage],
    measurements: np.ndarray,
    config: PluginConfig,
) -> FilterPass:
    """执行前向 Kalman 滤波。

    Args:
        observations: 观测消息序列。
        measurements: 观测向量序列。
        config: 插件配置。

    Returns:
        FilterPass: 前向滤波阶段的结果。
    """
    size = len(observations)
    identity = np.eye(4)
    observation_matrix = _observation_matrix()
    base_r = np.eye(2, dtype=float) * (config.measurement_sigma_m**2)

    filtered_states: list[np.ndarray] = []
    filtered_covariances: list[np.ndarray] = []
    predicted_states: list[np.ndarray | None] = [None] * size
    predicted_covariances: list[np.ndarray | None] = [None] * size
    transition_matrices: list[np.ndarray | None] = [None] * size

    state, covariance = _initial_state(measurements, config)
    filtered_states.append(state.copy())
    filtered_covariances.append(covariance.copy())

    for index in range(1, size):
        prediction = _predict_step(
            observations,
            state,
            covariance,
            index,
            config,
        )
        predicted_state, predicted_covariance, transition, dt = prediction
        residual = measurements[index] - (observation_matrix @ predicted_state)
        residual_speed = _residual_speed(residual, dt)
        _record_prediction(
            predicted_states,
            predicted_covariances,
            transition_matrices,
            index,
            predicted_state,
            predicted_covariance,
            transition,
        )
        state, covariance = _update_state(
            predicted_state,
            predicted_covariance,
            residual,
            residual_speed,
            observation_matrix,
            base_r,
            identity,
            config,
        )
        filtered_states.append(state.copy())
        filtered_covariances.append(covariance.copy())

    return FilterPass(
        filtered_states=filtered_states,
        filtered_covariances=filtered_covariances,
        predicted_states=predicted_states,
        predicted_covariances=predicted_covariances,
        transition_matrices=transition_matrices,
    )


def backward_smooth(filter_pass: FilterPass) -> list[np.ndarray]:
    """执行后向 RTS 平滑。

    Args:
        filter_pass: 前向滤波阶段的结果。

    Returns:
        list[np.ndarray]: RTS 反向平滑后的状态序列。
    """
    smoothed_states = [
        state.copy() for state in filter_pass.filtered_states
    ]
    smoothed_covariances = [
        covariance.copy()
        for covariance in filter_pass.filtered_covariances
    ]
    for index in range(len(filter_pass.filtered_states) - 2, -1, -1):
        next_transition = filter_pass.transition_matrices[index + 1]
        next_prediction = filter_pass.predicted_covariances[index + 1]
        next_state_prediction = filter_pass.predicted_states[index + 1]
        if _missing_prediction_inputs(
            next_transition,
            next_prediction,
            next_state_prediction,
        ):
            continue
        gain = _smoother_gain(
            filter_pass.filtered_covariances[index],
            next_transition,
            next_prediction,
        )
        smoothed_states[index] = _smooth_state(
            filter_pass.filtered_states[index],
            smoothed_states[index + 1],
            next_state_prediction,
            gain,
        )
        smoothed_covariances[index] = _smooth_covariance(
            filter_pass.filtered_covariances[index],
            smoothed_covariances[index + 1],
            next_prediction,
            gain,
        )
    return smoothed_states


def _observation_matrix() -> np.ndarray:
    """构造位置观测矩阵。

    Args:
        None. 不接收额外参数。

    Returns:
        np.ndarray: 位置观测矩阵。
    """
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=float,
    )


def _initial_state(
    measurements: np.ndarray,
    config: PluginConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """构造首个滤波状态与协方差。

    Args:
        measurements: 观测向量序列。
        config: 插件配置。

    Returns:
        tuple[np.ndarray, np.ndarray]: 初始状态和协方差。
    """
    state = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0])
    covariance = np.diag(
        [
            config.initial_position_sigma_m**2,
            config.initial_position_sigma_m**2,
            config.initial_velocity_sigma_mps**2,
            config.initial_velocity_sigma_mps**2,
        ]
    )
    return state, covariance


def _predict_step(
    observations: Sequence[ParsedMessage],
    state: np.ndarray,
    covariance: np.ndarray,
    index: int,
    config: PluginConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """执行单步状态预测。

    Args:
        observations: 观测消息序列。
        state: 当前滤波状态。
        covariance: 当前状态协方差。
        index: 目标元素索引。
        config: 插件配置。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            预测状态、预测协方差、状态转移矩阵和有效时间步长。
    """
    dt = effective_dt(observations[index - 1], observations[index], config)
    transition = transition_matrix(dt)
    predicted_state = transition @ state
    predicted_covariance = (
        transition @ covariance @ transition.T
        + process_noise(dt, config.process_accel_sigma_mps2)
    )
    return predicted_state, predicted_covariance, transition, dt


def _residual_speed(residual: np.ndarray, dt: float) -> float:
    """把位置残差换算成等效速度。

    Args:
        residual: 当前位置残差。
        dt: 有效时间步长。

    Returns:
        float: 等效残差速度。
    """
    return float(np.linalg.norm(residual) / dt)


def _record_prediction(
    predicted_states: list[np.ndarray | None],
    predicted_covariances: list[np.ndarray | None],
    transition_matrices: list[np.ndarray | None],
    index: int,
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    transition: np.ndarray,
) -> None:
    """记录单步预测结果。

    Args:
        predicted_states: 预测状态序列。
        predicted_covariances: 预测协方差序列。
        transition_matrices: 状态转移矩阵序列。
        index: 当前索引。
        predicted_state: 当前预测状态。
        predicted_covariance: 当前预测协方差。
        transition: 当前状态转移矩阵。

    Returns:
        None: 不返回值。
    """
    predicted_states[index] = predicted_state
    predicted_covariances[index] = predicted_covariance
    transition_matrices[index] = transition


def _update_state(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    residual: np.ndarray,
    residual_speed: float,
    observation_matrix: np.ndarray,
    base_r: np.ndarray,
    identity: np.ndarray,
    config: PluginConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """按残差大小更新状态。

    Args:
        predicted_state: 当前预测状态。
        predicted_covariance: 当前预测协方差。
        residual: 当前位置残差。
        residual_speed: 等效残差速度。
        observation_matrix: 观测矩阵。
        base_r: 基础观测噪声矩阵。
        identity: 单位矩阵。
        config: 插件配置。

    Returns:
        tuple[np.ndarray, np.ndarray]: 更新后的状态和协方差。
    """
    if residual_speed > config.hard_residual_speed_mps:
        return _apply_hard_reject(
            predicted_state,
            predicted_covariance,
            config,
        )
    return _apply_measurement_update(
        predicted_state,
        predicted_covariance,
        residual,
        residual_speed,
        observation_matrix,
        base_r,
        identity,
        config,
    )


def _apply_hard_reject(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    config: PluginConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """执行硬拒绝更新分支。

    Args:
        predicted_state: 当前预测状态。
        predicted_covariance: 当前预测协方差。
        config: 插件配置。

    Returns:
        tuple[np.ndarray, np.ndarray]: 硬拒绝后的状态和协方差。
    """
    state = predicted_state.copy()
    covariance = predicted_covariance.copy()
    if config.reset_velocity_on_reject:
        state[2:] = 0.0
        covariance[0:2, 2:] = 0.0
        covariance[2:, 0:2] = 0.0
        covariance[2:, 2:] = (
            np.eye(2, dtype=float) * (config.initial_velocity_sigma_mps**2)
        )
    return state, covariance


def _apply_measurement_update(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    residual: np.ndarray,
    residual_speed: float,
    observation_matrix: np.ndarray,
    base_r: np.ndarray,
    identity: np.ndarray,
    config: PluginConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """执行正常或软抑制更新分支。

    Args:
        predicted_state: 当前预测状态。
        predicted_covariance: 当前预测协方差。
        residual: 当前位置残差。
        residual_speed: 等效残差速度。
        observation_matrix: 观测矩阵。
        base_r: 基础观测噪声矩阵。
        identity: 单位矩阵。
        config: 插件配置。

    Returns:
        tuple[np.ndarray, np.ndarray]: 更新后的状态和协方差。
    """
    scale = (
        config.soft_noise_scale
        if residual_speed > config.soft_residual_speed_mps
        else 1.0
    )
    measurement_noise = base_r * scale
    innovation = (
        observation_matrix @ predicted_covariance @ observation_matrix.T
        + measurement_noise
    )
    gain = predicted_covariance @ observation_matrix.T @ np.linalg.inv(
        innovation
    )
    state = predicted_state + gain @ residual
    covariance = (identity - gain @ observation_matrix) @ predicted_covariance
    covariance = 0.5 * (covariance + covariance.T)
    return state, covariance


def _missing_prediction_inputs(
    transition: np.ndarray | None,
    prediction: np.ndarray | None,
    state_prediction: np.ndarray | None,
) -> bool:
    """判断反向平滑是否缺少必要输入。

    Args:
        transition: 下一步状态转移矩阵。
        prediction: 下一步预测协方差。
        state_prediction: 下一步预测状态。

    Returns:
        bool: 是否缺少任一必要输入。
    """
    return transition is None or prediction is None or state_prediction is None


def _smoother_gain(
    filtered_covariance: np.ndarray,
    transition: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """计算 RTS 平滑增益。

    Args:
        filtered_covariance: 当前时刻滤波协方差。
        transition: 下一步状态转移矩阵。
        prediction: 下一步预测协方差。

    Returns:
        np.ndarray: RTS 平滑增益。
    """
    return filtered_covariance @ transition.T @ np.linalg.inv(prediction)


def _smooth_state(
    filtered_state: np.ndarray,
    next_smoothed_state: np.ndarray,
    next_state_prediction: np.ndarray,
    gain: np.ndarray,
) -> np.ndarray:
    """执行单步 RTS 状态回代。

    Args:
        filtered_state: 当前时刻滤波状态。
        next_smoothed_state: 下一时刻平滑状态。
        next_state_prediction: 下一时刻预测状态。
        gain: RTS 平滑增益。

    Returns:
        np.ndarray: 当前时刻平滑状态。
    """
    return filtered_state + gain @ (next_smoothed_state - next_state_prediction)


def _smooth_covariance(
    filtered_covariance: np.ndarray,
    next_smoothed_covariance: np.ndarray,
    next_prediction: np.ndarray,
    gain: np.ndarray,
) -> np.ndarray:
    """执行单步 RTS 协方差回代。

    Args:
        filtered_covariance: 当前时刻滤波协方差。
        next_smoothed_covariance: 下一时刻平滑协方差。
        next_prediction: 下一时刻预测协方差。
        gain: RTS 平滑增益。

    Returns:
        np.ndarray: 当前时刻平滑协方差。
    """
    return (
        filtered_covariance
        + gain @ (next_smoothed_covariance - next_prediction) @ gain.T
    )
