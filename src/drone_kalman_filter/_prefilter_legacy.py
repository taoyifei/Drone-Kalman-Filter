"""legacy 路径下的单点异常修复逻辑。"""

from __future__ import annotations

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import ParsedMessage
from drone_kalman_filter._prefilter_geometry import (
    distance,
    dt_seconds,
    extrapolate_point,
    interpolate_point,
)
from drone_kalman_filter._prefilter_types import LocalPoint


def repair_points_legacy(
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
) -> list[LocalPoint]:
    """按旧路径修复孤立异常点。"""
    repaired = [raw_points[0]]

    for index in range(1, len(raw_points)):
        raw_point = raw_points[index]
        previous_point = repaired[index - 1]
        dt = dt_seconds(observations[index - 1], observations[index], config)
        distance_to_previous = distance(previous_point, raw_point)
        implied_speed = distance_to_previous / dt
        hard_jump = (implied_speed > config.prefilter_hard_speed_mps or
                     (dt <= config.prefilter_hard_distance_dt_seconds and
                      distance_to_previous > config.prefilter_hard_distance_m))
        bridge_spike = (implied_speed > config.prefilter_soft_speed_mps and
                        is_bridge_spike_legacy(
                            observations=observations,
                            raw_points=raw_points,
                            repaired_points=repaired,
                            index=index,
                            config=config,
                        ))

        if hard_jump or bridge_spike:
            repaired.append(
                repair_single_point_legacy(index, observations, raw_points,
                                           repaired, config))
            continue

        repaired.append(raw_point)

    return repaired


def repair_single_point_legacy(
    index: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    repaired_points: list[LocalPoint],
    config: PluginConfig,
) -> LocalPoint:
    """按旧路径修复一个孤立异常点。"""
    previous_point = repaired_points[index - 1]

    if index < len(raw_points) - 1:
        next_raw_point = raw_points[index + 1]
        if distance(previous_point,
                    next_raw_point) <= config.prefilter_hard_distance_m:
            return interpolate_point(
                left_point=previous_point,
                right_point=next_raw_point,
                left_time=observations[index - 1].event_time,
                current_time=observations[index].event_time,
                right_time=observations[index + 1].event_time,
            )

    if index >= 2:
        return extrapolate_point(
            left_point=repaired_points[index - 2],
            right_point=repaired_points[index - 1],
            left_time=observations[index - 2].event_time,
            right_time=observations[index - 1].event_time,
            current_time=observations[index].event_time,
            config=config,
        )

    return previous_point


def is_bridge_spike_legacy(
    *,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    repaired_points: list[LocalPoint],
    index: int,
    config: PluginConfig,
) -> bool:
    """判断当前点是否符合旧路径里的孤立尖刺模式。"""
    if index >= len(raw_points) - 1:
        return False

    previous_point = repaired_points[index - 1]
    current_point = raw_points[index]
    next_point = raw_points[index + 1]
    neighbor_distance = distance(previous_point, next_point)
    current_to_previous = distance(current_point, previous_point)
    current_to_next = distance(current_point, next_point)
    if neighbor_distance > config.prefilter_bridge_neighbor_distance_m:
        return False
    if min(current_to_previous,
           current_to_next) <= config.prefilter_bridge_center_distance_m:
        return False

    left_dt = dt_seconds(observations[index - 1], observations[index], config)
    right_dt = dt_seconds(observations[index], observations[index + 1], config)
    return (left_dt <= config.prefilter_hard_distance_dt_seconds and
            right_dt <= config.prefilter_hard_distance_dt_seconds)
