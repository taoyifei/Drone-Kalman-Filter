"""可疑点分类与单点修复逻辑。"""

from __future__ import annotations

from datetime import datetime

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import ParsedMessage
from drone_kalman_filter._prefilter_geometry import (
    distance,
    dot,
    dt_seconds,
    extrapolate_point,
    interpolate_point,
    vector,
    vector_norm,
)
from drone_kalman_filter._prefilter_types import (
    LocalPoint,
    Suspicion,
    TrustedAnchor,
)


def has_burst_candidate(
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    seed_anchor: TrustedAnchor | None,
) -> bool:
    """判断当前窗口是否值得切到 burst 修复路径。

    Args:
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        bool: 当前窗口是否存在 burst 修复候选。
    """
    if len(raw_points) < 3:
        return False

    last_trusted_point, last_trusted_time, index = _initial_trusted_anchor(
        observations,
        raw_points,
        seed_anchor,
    )

    while index < len(raw_points):
        suspicion = classify_suspicion(
            anchor_point=last_trusted_point,
            anchor_time=last_trusted_time,
            index=index,
            observations=observations,
            raw_points=raw_points,
            config=config,
        )
        if not suspicion.is_suspicious:
            last_trusted_point = raw_points[index]
            last_trusted_time = observations[index].event_time
            index += 1
            continue

        run_end = _extend_suspicious_run(
            start_index=index,
            anchor_point=last_trusted_point,
            anchor_time=last_trusted_time,
            observations=observations,
            raw_points=raw_points,
            config=config,
        )
        if _qualifies_candidate_run(
            anchor_point=last_trusted_point,
            raw_points=raw_points,
            start_index=index,
            run_end=run_end,
        ):
            return True

        index += 1

    return False


def _initial_trusted_anchor(
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    seed_anchor: TrustedAnchor | None,
) -> tuple[LocalPoint, datetime, int]:
    """构造 burst 候选扫描的初始可信锚点。

    Args:
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        tuple[LocalPoint, datetime, int]:
            可信锚点、锚点时间和扫描起始索引。
    """
    if seed_anchor is None:
        return raw_points[0], observations[0].event_time, 1
    return seed_anchor.point, seed_anchor.event_time, 0


def _extend_suspicious_run(
    *,
    start_index: int,
    anchor_point: LocalPoint,
    anchor_time: datetime,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
) -> int:
    """扩展当前连续可疑 run 的结束位置。

    Args:
        start_index: 连续可疑区间的起始索引。
        anchor_point: 当前可信锚点。
        anchor_time: 当前可信锚点时间。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。

    Returns:
        int: 连续可疑区间的结束索引。
    """
    run_end = start_index
    while (
        run_end + 1 < len(raw_points)
        and run_end - start_index + 1 < config.prefilter_burst_max_run_length
    ):
        local_dt = dt_seconds(
            observations[run_end],
            observations[run_end + 1],
            config,
        )
        if local_dt > config.prefilter_hard_distance_dt_seconds:
            break
        next_suspicion = classify_suspicion(
            anchor_point=anchor_point,
            anchor_time=anchor_time,
            index=run_end + 1,
            observations=observations,
            raw_points=raw_points,
            config=config,
        )
        if not next_suspicion.is_suspicious:
            break
        run_end += 1
    return run_end


def _qualifies_candidate_run(
    *,
    anchor_point: LocalPoint,
    raw_points: list[LocalPoint],
    start_index: int,
    run_end: int,
) -> bool:
    """判断当前连续可疑 run 是否构成 burst 候选。

    Args:
        anchor_point: 当前可信锚点。
        raw_points: 当前窗口内的局部平面原始点序列。
        start_index: 连续可疑区间的起始索引。
        run_end: 连续可疑区间的结束索引。

    Returns:
        bool: 是否构成 burst 候选。
    """
    return run_end > start_index and qualifies_as_burst(
        anchor_point=anchor_point,
        raw_points=raw_points,
        start_index=start_index,
        run_end=run_end,
    )


def qualifies_as_burst(
    *,
    anchor_point: LocalPoint,
    raw_points: list[LocalPoint],
    start_index: int,
    run_end: int,
) -> bool:
    """判断一串连续可疑点是否满足 burst 条件。

    Args:
        anchor_point: 用于判定和修复的参考锚点。
        raw_points: 当前窗口内的局部平面原始点序列。
        start_index: 连续可疑区间的起始索引。
        run_end: 连续可疑区间的结束索引。

    Returns:
        bool: 指定区间是否满足 burst 条件。
    """
    if run_end <= start_index:
        return False

    first_vector = vector(anchor_point, raw_points[start_index])
    first_norm = vector_norm(first_vector)
    if first_norm == 0.0:
        return False

    for current_index in range(start_index + 1, run_end + 1):
        current_vector = vector(anchor_point, raw_points[current_index])
        current_norm = vector_norm(current_vector)
        if current_norm == 0.0:
            continue
        cosine = dot(first_vector, current_vector) / (
            first_norm * current_norm
        )
        if cosine < -0.5:
            return True
    return False


def repair_single_point(
    index: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    repaired_points: list[LocalPoint],
    repaired_times: list[datetime],
    anchor_point: LocalPoint,
    anchor_time: datetime,
    config: PluginConfig,
) -> LocalPoint:
    """按单点路径修复一个独立的异常点。

    Args:
        index: 目标点在窗口内的索引。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        repaired_points: 当前已修复的点序列。
        repaired_times: 当前已修复点的时间序列。
        anchor_point: 用于判定和修复的参考锚点。
        anchor_time: 参考锚点对应的事件时间。
        config: 插件配置。

    Returns:
        LocalPoint: 单个可疑点的修复结果。
    """
    if index < len(raw_points) - 1:
        next_raw_point = raw_points[index + 1]
        if distance(anchor_point, next_raw_point) <= (
            config.prefilter_hard_distance_m
        ):
            return interpolate_point(
                left_point=anchor_point,
                right_point=next_raw_point,
                left_time=anchor_time,
                current_time=observations[index].event_time,
                right_time=observations[index + 1].event_time,
            )

    if repaired_points:
        left_point = repaired_points[-2] if len(
            repaired_points
        ) >= 2 else anchor_point
        left_time = repaired_times[-2] if len(
            repaired_times
        ) >= 2 else anchor_time
        return extrapolate_point(
            left_point=left_point,
            right_point=repaired_points[-1],
            left_time=left_time,
            right_time=repaired_times[-1],
            current_time=observations[index].event_time,
            config=config,
        )

    return anchor_point


def classify_suspicion(
    *,
    anchor_point: LocalPoint,
    anchor_time: datetime,
    index: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
) -> Suspicion:
    """相对可信锚点给当前点做异常分类。

    Args:
        anchor_point: 用于判定和修复的参考锚点。
        anchor_time: 参考锚点对应的事件时间。
        index: 目标点在窗口内的索引。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。

    Returns:
        Suspicion: 当前点的可疑性分类结果。
    """
    raw_point = raw_points[index]
    dt = max(
        (observations[index].event_time - anchor_time).total_seconds(),
        config.min_dt_seconds,
    )
    distance_to_anchor = distance(anchor_point, raw_point)
    implied_speed = distance_to_anchor / dt
    hard_jump = (
        implied_speed > config.prefilter_hard_speed_mps
        or (
            dt <= config.prefilter_hard_distance_dt_seconds
            and distance_to_anchor > config.prefilter_hard_distance_m
        )
    )
    bridge_spike = (
        implied_speed > config.prefilter_soft_speed_mps
        and is_bridge_spike_from_anchor(
            observations=observations,
            raw_points=raw_points,
            anchor_point=anchor_point,
            anchor_time=anchor_time,
            index=index,
            config=config,
        )
    )
    return Suspicion(hard_jump=hard_jump, bridge_spike=bridge_spike)


def is_bridge_spike_from_anchor(
    *,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    anchor_point: LocalPoint,
    anchor_time: datetime,
    index: int,
    config: PluginConfig,
) -> bool:
    """判断当前点相对可信锚点是否构成桥接尖刺。

    Args:
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        anchor_point: 用于判定和修复的参考锚点。
        anchor_time: 参考锚点对应的事件时间。
        index: 目标点在窗口内的索引。
        config: 插件配置。

    Returns:
        bool: 当前点是否构成基于锚点判定的桥接尖刺。
    """
    if index >= len(raw_points) - 1:
        return False

    current_point = raw_points[index]
    next_point = raw_points[index + 1]
    neighbor_distance = distance(anchor_point, next_point)
    current_to_previous = distance(current_point, anchor_point)
    current_to_next = distance(current_point, next_point)
    if neighbor_distance > config.prefilter_bridge_neighbor_distance_m:
        return False
    if min(
        current_to_previous,
        current_to_next,
    ) <= config.prefilter_bridge_center_distance_m:
        return False

    left_dt = max(
        (observations[index].event_time - anchor_time).total_seconds(),
        config.min_dt_seconds,
    )
    right_dt = dt_seconds(observations[index], observations[index + 1], config)
    return (
        left_dt <= config.prefilter_hard_distance_dt_seconds
        and right_dt <= config.prefilter_hard_distance_dt_seconds
    )
