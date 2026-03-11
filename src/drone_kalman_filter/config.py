from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class PluginConfig:
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
        # 这里集中做参数防御，避免运行时出现难排查的配置组合错误。
        if self.smoother_mode not in {"kalman", "robust_prefilter_kalman"}:
            raise ValueError("smoother_mode must be 'kalman' or 'robust_prefilter_kalman'")
        if self.window_size < 3:
            raise ValueError("window_size must be at least 3")
        if self.lag_points < 1:
            raise ValueError("lag_points must be positive")
        if self.lag_points >= self.window_size:
            raise ValueError("lag_points must be smaller than window_size")
        if self.min_dt_seconds <= 0:
            raise ValueError("min_dt_seconds must be positive")
        if self.max_segment_gap_seconds <= 0:
            raise ValueError("max_segment_gap_seconds must be positive")
        if self.idle_flush_seconds <= 0:
            raise ValueError("idle_flush_seconds must be positive")
        if self.measurement_sigma_m <= 0:
            raise ValueError("measurement_sigma_m must be positive")
        if self.process_accel_sigma_mps2 <= 0:
            raise ValueError("process_accel_sigma_mps2 must be positive")
        if self.initial_position_sigma_m <= 0:
            raise ValueError("initial_position_sigma_m must be positive")
        if self.initial_velocity_sigma_mps <= 0:
            raise ValueError("initial_velocity_sigma_mps must be positive")
        if self.soft_residual_speed_mps <= 0:
            raise ValueError("soft_residual_speed_mps must be positive")
        if self.hard_residual_speed_mps <= self.soft_residual_speed_mps:
            raise ValueError("hard_residual_speed_mps must be larger than soft_residual_speed_mps")
        if self.soft_noise_scale < 1.0:
            raise ValueError("soft_noise_scale must be at least 1.0")
        if self.prefilter_window_size < 3:
            raise ValueError("prefilter_window_size must be at least 3")
        if self.prefilter_lag_points < 1:
            raise ValueError("prefilter_lag_points must be positive")
        if self.prefilter_lag_points >= self.prefilter_window_size:
            raise ValueError("prefilter_lag_points must be smaller than prefilter_window_size")
        if self.prefilter_soft_speed_mps <= 0:
            raise ValueError("prefilter_soft_speed_mps must be positive")
        if self.prefilter_hard_speed_mps <= self.prefilter_soft_speed_mps:
            raise ValueError("prefilter_hard_speed_mps must be larger than prefilter_soft_speed_mps")
        if self.prefilter_hard_distance_m <= 0:
            raise ValueError("prefilter_hard_distance_m must be positive")
        if self.prefilter_hard_distance_dt_seconds <= 0:
            raise ValueError("prefilter_hard_distance_dt_seconds must be positive")
        if self.prefilter_bridge_neighbor_distance_m <= 0:
            raise ValueError("prefilter_bridge_neighbor_distance_m must be positive")
        if self.prefilter_bridge_center_distance_m <= self.prefilter_bridge_neighbor_distance_m:
            raise ValueError("prefilter_bridge_center_distance_m must be larger than prefilter_bridge_neighbor_distance_m")
        if self.prefilter_burst_max_run_length < 1:
            raise ValueError("prefilter_burst_max_run_length must be at least 1")
        if self.prefilter_median_window_size < 1 or self.prefilter_median_window_size % 2 == 0:
            raise ValueError("prefilter_median_window_size must be a positive odd integer")
