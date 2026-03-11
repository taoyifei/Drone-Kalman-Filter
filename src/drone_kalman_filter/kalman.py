from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import PluginConfig
from .geo import LocalTangentPlane
from .message import ParsedMessage


@dataclass(frozen=True, slots=True)
class SmoothedPosition:
    east_m: float
    north_m: float


def smooth_positions(
    observations: Sequence[ParsedMessage],
    plane: LocalTangentPlane,
    config: PluginConfig,
) -> list[SmoothedPosition]:
    if not observations:
        return []

    if len(observations) == 1:
        east, north = plane.to_local(observations[0].latitude, observations[0].longitude)
        return [SmoothedPosition(east, north)]

    measurements = np.array(
        [plane.to_local(item.latitude, item.longitude) for item in observations],
        dtype=float,
    )
    # 二维恒速模型：[east, north, ve, vn]，观测量只有位置。
    size = len(observations)
    identity = np.eye(4)
    observation_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    base_r = np.eye(2, dtype=float) * (config.measurement_sigma_m**2)

    filtered_states: list[np.ndarray] = []
    filtered_covariances: list[np.ndarray] = []
    predicted_states: list[np.ndarray | None] = [None] * size
    predicted_covariances: list[np.ndarray | None] = [None] * size
    transition_matrices: list[np.ndarray | None] = [None] * size

    state = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0], dtype=float)
    covariance = np.diag(
        [
            config.initial_position_sigma_m**2,
            config.initial_position_sigma_m**2,
            config.initial_velocity_sigma_mps**2,
            config.initial_velocity_sigma_mps**2,
        ]
    )
    filtered_states.append(state.copy())
    filtered_covariances.append(covariance.copy())

    for index in range(1, size):
        dt = _effective_dt(observations[index - 1], observations[index], config)
        transition = _transition_matrix(dt)
        process_noise = _process_noise(dt, config.process_accel_sigma_mps2)

        predicted_state = transition @ state
        predicted_covariance = transition @ covariance @ transition.T + process_noise

        residual = measurements[index] - (observation_matrix @ predicted_state)
        residual_speed = float(np.linalg.norm(residual) / dt)

        predicted_states[index] = predicted_state
        predicted_covariances[index] = predicted_covariance
        transition_matrices[index] = transition

        # 观测残差过大时，认为这一帧不可信，直接跳过更新，避免把状态带飞。
        if residual_speed > config.hard_residual_speed_mps:
            state = predicted_state.copy()
            covariance = predicted_covariance.copy()
            if config.reset_velocity_on_reject:
                # baseline 诊断链路可以开启这项，把受污染的速度状态直接砍掉。
                state[2:] = 0.0
                covariance[0:2, 2:] = 0.0
                covariance[2:, 0:2] = 0.0
                covariance[2:, 2:] = np.eye(2, dtype=float) * (config.initial_velocity_sigma_mps**2)
        else:
            # 软异常不直接拒绝，而是放大观测噪声，弱化这一帧的影响。
            scale = config.soft_noise_scale if residual_speed > config.soft_residual_speed_mps else 1.0
            measurement_noise = base_r * scale
            innovation = observation_matrix @ predicted_covariance @ observation_matrix.T + measurement_noise
            gain = predicted_covariance @ observation_matrix.T @ np.linalg.inv(innovation)
            state = predicted_state + gain @ residual
            covariance = (identity - gain @ observation_matrix) @ predicted_covariance
            covariance = 0.5 * (covariance + covariance.T)

        filtered_states.append(state.copy())
        filtered_covariances.append(covariance.copy())

    smoothed_states = [state.copy() for state in filtered_states]
    smoothed_covariances = [covariance.copy() for covariance in filtered_covariances]

    # 后向 RTS 回算：用有限未来信息修正过去位置，减少固定滞后输出的锯齿感。
    for index in range(size - 2, -1, -1):
        next_transition = transition_matrices[index + 1]
        next_prediction = predicted_covariances[index + 1]
        next_state_prediction = predicted_states[index + 1]
        if next_transition is None or next_prediction is None or next_state_prediction is None:
            continue
        smoother_gain = filtered_covariances[index] @ next_transition.T @ np.linalg.inv(next_prediction)
        smoothed_states[index] = filtered_states[index] + smoother_gain @ (
            smoothed_states[index + 1] - next_state_prediction
        )
        smoothed_covariances[index] = filtered_covariances[index] + smoother_gain @ (
            smoothed_covariances[index + 1] - next_prediction
        ) @ smoother_gain.T

    return [SmoothedPosition(float(state[0]), float(state[1])) for state in smoothed_states]


def _effective_dt(previous: ParsedMessage, current: ParsedMessage, config: PluginConfig) -> float:
    delta = (current.event_time - previous.event_time).total_seconds()
    return max(delta, config.min_dt_seconds)


def _transition_matrix(dt: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _process_noise(dt: float, sigma_acceleration: float) -> np.ndarray:
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
