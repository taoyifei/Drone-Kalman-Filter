"""前向 Kalman 与后向 RTS 的内部实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import ParsedMessage
from drone_kalman_filter._kalman_model import (
    effective_dt,
    process_noise,
    transition_matrix,
)


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
    """执行前向 Kalman 滤波。"""
    size = len(observations)
    identity = np.eye(4)
    observation_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                                  dtype=float)
    base_r = np.eye(2, dtype=float) * (config.measurement_sigma_m**2)

    filtered_states: list[np.ndarray] = []
    filtered_covariances: list[np.ndarray] = []
    predicted_states: list[np.ndarray | None] = [None] * size
    predicted_covariances: list[np.ndarray | None] = [None] * size
    transition_matrices: list[np.ndarray | None] = [None] * size

    state = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0],
                     dtype=float)
    covariance = np.diag([
        config.initial_position_sigma_m**2,
        config.initial_position_sigma_m**2,
        config.initial_velocity_sigma_mps**2,
        config.initial_velocity_sigma_mps**2,
    ])
    filtered_states.append(state.copy())
    filtered_covariances.append(covariance.copy())

    for index in range(1, size):
        dt = effective_dt(observations[index - 1], observations[index], config)
        transition = transition_matrix(dt)
        predicted_state = transition @ state
        predicted_covariance = (
            transition @ covariance @ transition.T +
            process_noise(dt, config.process_accel_sigma_mps2))
        residual = measurements[index] - (observation_matrix @ predicted_state)
        residual_speed = float(np.linalg.norm(residual) / dt)

        predicted_states[index] = predicted_state
        predicted_covariances[index] = predicted_covariance
        transition_matrices[index] = transition

        if residual_speed > config.hard_residual_speed_mps:
            state = predicted_state.copy()
            covariance = predicted_covariance.copy()
            if config.reset_velocity_on_reject:
                state[2:] = 0.0
                covariance[0:2, 2:] = 0.0
                covariance[2:, 0:2] = 0.0
                covariance[2:, 2:] = np.eye(
                    2, dtype=float) * (config.initial_velocity_sigma_mps**2)
        else:
            scale = (config.soft_noise_scale if residual_speed
                     > config.soft_residual_speed_mps else 1.0)
            measurement_noise = base_r * scale
            innovation = (
                observation_matrix @ predicted_covariance @ observation_matrix.T
                + measurement_noise)
            gain = predicted_covariance @ observation_matrix.T @ np.linalg.inv(
                innovation)
            state = predicted_state + gain @ residual
            covariance = (identity -
                          gain @ observation_matrix) @ predicted_covariance
            covariance = 0.5 * (covariance + covariance.T)

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
    """执行后向 RTS 平滑。"""
    size = len(filter_pass.filtered_states)
    smoothed_states = [state.copy() for state in filter_pass.filtered_states]
    smoothed_covariances = [
        covariance.copy() for covariance in filter_pass.filtered_covariances
    ]

    for index in range(size - 2, -1, -1):
        next_transition = filter_pass.transition_matrices[index + 1]
        next_prediction = filter_pass.predicted_covariances[index + 1]
        next_state_prediction = filter_pass.predicted_states[index + 1]
        if (next_transition is None or next_prediction is None or
                next_state_prediction is None):
            continue
        smoother_gain = (filter_pass.filtered_covariances[index]
                         @ next_transition.T @ np.linalg.inv(next_prediction))
        smoothed_states[
            index] = filter_pass.filtered_states[index] + smoother_gain @ (
                smoothed_states[index + 1] - next_state_prediction)
        smoothed_covariances[index] = filter_pass.filtered_covariances[
            index] + smoother_gain @ (smoothed_covariances[index + 1] -
                                      next_prediction) @ smoother_gain.T

    return smoothed_states
