"""主链路配置与配置校验。"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class PluginConfig:
    """保存主链路运行需要的全部配置。"""

    # 主链路模式：纯 Kalman，或“预处理 + Kalman”的鲁棒模式。
    smoother_mode: Literal["kalman", "robust_prefilter_kalman"] = "kalman"
    # 主平滑窗口参数。
    window_size: int = 5
    lag_points: int = 2
    min_dt_seconds: float = 0.3
    max_segment_gap_seconds: float = 10.0
    idle_flush_seconds: float = 10.0
    # Kalman 模型参数。
    measurement_sigma_m: float = 12.0
    process_accel_sigma_mps2: float = 6.0
    initial_position_sigma_m: float = 25.0
    initial_velocity_sigma_mps: float = 20.0
    soft_residual_speed_mps: float = 25.0
    hard_residual_speed_mps: float = 35.0
    soft_noise_scale: float = 9.0
    # 仅供离线 baseline 使用：硬拒绝后是否直接清零速度状态。
    reset_velocity_on_reject: bool = False
    # 预处理层参数。
    prefilter_window_size: int = 5
    prefilter_lag_points: int = 2
    prefilter_soft_speed_mps: float = 25.0
    prefilter_hard_speed_mps: float = 45.0
    prefilter_hard_distance_m: float = 80.0
    prefilter_hard_distance_dt_seconds: float = 1.5
    prefilter_bridge_neighbor_distance_m: float = 30.0
    prefilter_bridge_center_distance_m: float = 50.0
    prefilter_burst_max_run_length: int = 3
    prefilter_median_window_size: int = 3

    def __post_init__(self) -> None:
        """校验配置项组合是否合法。

        Args:
            None. 不接收额外参数。

        Returns:
            None: 不返回值。

        Raises:
            ValueError: 当配置组合非法或参数超出约束时抛出。
        """
        # 这里集中做参数防御，避免运行时出现难排查的配置组合错误。
        if self.smoother_mode not in {"kalman", "robust_prefilter_kalman"}:
            raise ValueError(
                "smoother_mode must be 'kalman' or 'robust_prefilter_kalman'")
        _validate_minimum("window_size",
                          self.window_size,
                          minimum=3,
                          message="window_size must be at least 3")
        _validate_positive("lag_points", self.lag_points)
        _validate_positive("min_dt_seconds", self.min_dt_seconds)
        _validate_positive("max_segment_gap_seconds",
                           self.max_segment_gap_seconds)
        _validate_positive("idle_flush_seconds", self.idle_flush_seconds)
        _validate_positive("measurement_sigma_m", self.measurement_sigma_m)
        _validate_positive("process_accel_sigma_mps2",
                           self.process_accel_sigma_mps2)
        _validate_positive("initial_position_sigma_m",
                           self.initial_position_sigma_m)
        _validate_positive("initial_velocity_sigma_mps",
                           self.initial_velocity_sigma_mps)
        _validate_positive("soft_residual_speed_mps",
                           self.soft_residual_speed_mps)
        if self.hard_residual_speed_mps <= self.soft_residual_speed_mps:
            raise ValueError("hard_residual_speed_mps must be larger than "
                             "soft_residual_speed_mps")
        _validate_minimum("soft_noise_scale",
                          self.soft_noise_scale,
                          minimum=1.0,
                          message="soft_noise_scale must be at least 1.0")
        _validate_minimum(
            "prefilter_window_size",
            self.prefilter_window_size,
            minimum=3,
            message="prefilter_window_size must be at least 3",
        )
        _validate_positive("prefilter_lag_points", self.prefilter_lag_points)
        _validate_positive("prefilter_soft_speed_mps",
                           self.prefilter_soft_speed_mps)
        if self.prefilter_hard_speed_mps <= self.prefilter_soft_speed_mps:
            raise ValueError("prefilter_hard_speed_mps must be larger than "
                             "prefilter_soft_speed_mps")
        _validate_positive("prefilter_hard_distance_m",
                           self.prefilter_hard_distance_m)
        _validate_positive("prefilter_hard_distance_dt_seconds",
                           self.prefilter_hard_distance_dt_seconds)
        _validate_positive("prefilter_bridge_neighbor_distance_m",
                           self.prefilter_bridge_neighbor_distance_m)
        if (self.prefilter_bridge_center_distance_m
                <= self.prefilter_bridge_neighbor_distance_m):
            raise ValueError(
                "prefilter_bridge_center_distance_m must be larger than "
                "prefilter_bridge_neighbor_distance_m")
        _validate_minimum(
            "prefilter_burst_max_run_length",
            self.prefilter_burst_max_run_length,
            minimum=1,
            message="prefilter_burst_max_run_length must be at least 1",
        )
        if (self.prefilter_median_window_size < 1 or
                self.prefilter_median_window_size % 2 == 0):
            raise ValueError(
                "prefilter_median_window_size must be a positive odd integer")
        _validate_window_relation("lag_points", self.lag_points, "window_size",
                                  self.window_size)
        _validate_window_relation(
            "prefilter_lag_points",
            self.prefilter_lag_points,
            "prefilter_window_size",
            self.prefilter_window_size,
        )


def _validate_positive(name: str, value: float) -> None:
    """校验数值必须为正。

    Args:
        name: 配置项名称。
        value: 输入值。

    Returns:
        None: 不返回值。

    Raises:
        ValueError: 当参数值不是正数时抛出。
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_minimum(name: str, value: float, *, minimum: float,
                      message: str) -> None:
    """校验数值必须不小于给定下限。

    Args:
        name: 配置项名称。
        value: 输入值。
        minimum: 允许的最小值。
        message: 校验失败时抛出的错误消息。

    Returns:
        None: 不返回值。

    Raises:
        ValueError: 当参数值小于允许的最小值时抛出。
    """
    if value < minimum:
        raise ValueError(message)


def _validate_window_relation(lag_name: str, lag_value: int, window_name: str,
                              window_value: int) -> None:
    """校验滞后点数必须小于窗口大小。

    Args:
        lag_name: 滞后参数名称。
        lag_value: 滞后参数值。
        window_name: 窗口参数名称。
        window_value: 窗口参数值。

    Returns:
        None: 不返回值。

    Raises:
        ValueError: 当滞后窗口不小于主窗口时抛出。
    """
    if lag_value >= window_value:
        raise ValueError(f"{lag_name} must be smaller than {window_name}")
