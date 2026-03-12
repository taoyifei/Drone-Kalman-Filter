"""burst 路径下的连续异常修复逻辑。"""

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
    BurstAnchor,
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
        observations: 观测消息序列。
        raw_points: 原始局部坐标点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续的可信锚点。

    Returns:
        bool: 当前窗口是否存在 burst 修复候选。
    """
    if len(raw_points) < 3:
        return False

    if seed_anchor is None:
        last_trusted_point = raw_points[0]
        last_trusted_time = observations[0].event_time
        index = 1
    else:
        last_trusted_point = seed_anchor.point
        last_trusted_time = seed_anchor.event_time
        index = 0

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

        run_end = index
        while (run_end + 1 < len(raw_points) and
               run_end - index + 1 < config.prefilter_burst_max_run_length):
            local_dt = dt_seconds(observations[run_end],
                                  observations[run_end + 1], config)
            if local_dt > config.prefilter_hard_distance_dt_seconds:
                break
            next_suspicion = classify_suspicion(
                anchor_point=last_trusted_point,
                anchor_time=last_trusted_time,
                index=run_end + 1,
                observations=observations,
                raw_points=raw_points,
                config=config,
            )
            if not next_suspicion.is_suspicious:
                break
            run_end += 1

        if run_end > index and qualifies_as_burst(
                anchor_point=last_trusted_point,
                raw_points=raw_points,
                start_index=index,
                run_end=run_end,
        ):
            return True

        index += 1

    return False


def repair_points_burst(
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    *,
    seed_anchor: TrustedAnchor | None = None,
) -> tuple[list[LocalPoint], bool, list[bool]]:
    """按 burst 规则修复连续异常点。

    Args:
        observations: 观测消息序列。
        raw_points: 原始局部坐标点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续的可信锚点。

    Returns:
        tuple[list[LocalPoint], bool, list[bool]]: burst 路径下的修复结果、是否发生修复以及改动标记。
    """
    repaired: list[LocalPoint] = []
    repaired_times: list[datetime] = []
    altered_flags: list[bool] = []
    had_burst_repair = False
    if seed_anchor is None:
        repaired.append(raw_points[0])
        repaired_times.append(observations[0].event_time)
        altered_flags.append(False)
        last_trusted_point = raw_points[0]
        last_trusted_time = observations[0].event_time
        index = 1
    else:
        last_trusted_point = seed_anchor.point
        last_trusted_time = seed_anchor.event_time
        index = 0

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
            repaired.append(raw_points[index])
            repaired_times.append(observations[index].event_time)
            altered_flags.append(False)
            last_trusted_point = raw_points[index]
            last_trusted_time = observations[index].event_time
            index += 1
            continue

        run_end = index
        while (run_end + 1 < len(raw_points) and
               run_end - index + 1 < config.prefilter_burst_max_run_length):
            local_dt = dt_seconds(observations[run_end],
                                  observations[run_end + 1], config)
            if local_dt > config.prefilter_hard_distance_dt_seconds:
                break
            next_suspicion = classify_suspicion(
                anchor_point=last_trusted_point,
                anchor_time=last_trusted_time,
                index=run_end + 1,
                observations=observations,
                raw_points=raw_points,
                config=config,
            )
            if not next_suspicion.is_suspicious:
                break
            run_end += 1

        is_burst = run_end > index and qualifies_as_burst(
            anchor_point=last_trusted_point,
            raw_points=raw_points,
            start_index=index,
            run_end=run_end,
        )
        if not is_burst:
            repaired_point = repair_single_point(
                index,
                observations,
                raw_points,
                repaired,
                repaired_times,
                last_trusted_point,
                last_trusted_time,
                config,
            )
            repaired.append(repaired_point)
            repaired_times.append(observations[index].event_time)
            altered_flags.append(True)
            last_trusted_point = repaired_point
            last_trusted_time = observations[index].event_time
            index += 1
            continue

        anchor = find_future_anchor(
            start_index=index,
            current_run_end=run_end,
            anchor_point=last_trusted_point,
            anchor_time=last_trusted_time,
            observations=observations,
            raw_points=raw_points,
            config=config,
        )

        if anchor.anchor_index is None:
            for current_index in range(index, anchor.repair_end + 1):
                repaired.append(last_trusted_point)
                repaired_times.append(observations[current_index].event_time)
                altered_flags.append(True)
            had_burst_repair = True
            index = anchor.repair_end + 1
            continue

        for current_index in range(index, anchor.anchor_index):
            repaired.append(
                interpolate_point(
                    left_point=last_trusted_point,
                    right_point=raw_points[anchor.anchor_index],
                    left_time=last_trusted_time,
                    current_time=observations[current_index].event_time,
                    right_time=observations[anchor.anchor_index].event_time,
                ))
            repaired_times.append(observations[current_index].event_time)
            altered_flags.append(True)
        repaired.append(raw_points[anchor.anchor_index])
        had_burst_repair = True
        repaired_times.append(observations[anchor.anchor_index].event_time)
        altered_flags.append(False)
        last_trusted_point = raw_points[anchor.anchor_index]
        last_trusted_time = observations[anchor.anchor_index].event_time
        index = anchor.anchor_index + 1

    return repaired, had_burst_repair, altered_flags


def find_future_anchor(
    *,
    start_index: int,
    current_run_end: int,
    anchor_point: LocalPoint,
    anchor_time: datetime,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
) -> BurstAnchor:
    """在窗口内寻找可用于 burst 修复的未来稳定锚点。

    Args:
        start_index: 连续可疑区间的起始索引。
        current_run_end: 当前连续可疑区间的结束索引。
        anchor_point: 用于判断或修复的参考锚点。
        anchor_time: 参考锚点对应的事件时间。
        observations: 观测消息序列。
        raw_points: 原始局部坐标点序列。
        config: 插件配置。

    Returns:
        BurstAnchor: 为 burst 修复找到的未来锚点。
    """
    repair_end = current_run_end
    search_limit = min(
        len(raw_points) - 1,
        start_index + config.prefilter_burst_max_run_length)

    for candidate_index in range(current_run_end + 1, search_limit + 1):
        candidate_suspicion = classify_suspicion(
            anchor_point=anchor_point,
            anchor_time=anchor_time,
            index=candidate_index,
            observations=observations,
            raw_points=raw_points,
            config=config,
        )
        if candidate_suspicion.is_suspicious:
            repair_end = candidate_index
            continue

        if candidate_index < len(raw_points) - 1:
            next_dt = dt_seconds(observations[candidate_index],
                                 observations[candidate_index + 1], config)
            next_distance = distance(raw_points[candidate_index],
                                     raw_points[candidate_index + 1])
            next_speed = next_distance / next_dt
            if next_speed > config.prefilter_hard_speed_mps or (
                    next_dt <= config.prefilter_hard_distance_dt_seconds and
                    next_distance > config.prefilter_hard_distance_m):
                repair_end = candidate_index
                continue

        return BurstAnchor(repair_end=repair_end, anchor_index=candidate_index)

    return BurstAnchor(repair_end=repair_end, anchor_index=None)


def qualifies_as_burst(
    *,
    anchor_point: LocalPoint,
    raw_points: list[LocalPoint],
    start_index: int,
    run_end: int,
) -> bool:
    """判断一串连续可疑点是否满足 burst 条件。

    Args:
        anchor_point: 用于判断或修复的参考锚点。
        raw_points: 原始局部坐标点序列。
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
        cosine = dot(first_vector, current_vector) / (first_norm * current_norm)
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
    """按新路径修复一个单独的异常点。

    Args:
        index: 目标元素在序列或缓冲区中的索引。
        observations: 观测消息序列。
        raw_points: 原始局部坐标点序列。
        repaired_points: 修复后的局部坐标点序列。
        repaired_times: 修复点对应的时间序列。
        anchor_point: 用于判断或修复的参考锚点。
        anchor_time: 参考锚点对应的事件时间。
        config: 插件配置。

    Returns:
        LocalPoint: 单个可疑点的修复结果。
    """
    if index < len(raw_points) - 1:
        next_raw_point = raw_points[index + 1]
        if distance(anchor_point,
                    next_raw_point) <= config.prefilter_hard_distance_m:
            return interpolate_point(
                left_point=anchor_point,
                right_point=next_raw_point,
                left_time=anchor_time,
                current_time=observations[index].event_time,
                right_time=observations[index + 1].event_time,
            )

    if repaired_points:
        left_point = repaired_points[-2] if len(
            repaired_points) >= 2 else anchor_point
        left_time = repaired_times[-2] if len(
            repaired_times) >= 2 else anchor_time
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
        anchor_point: 用于判断或修复的参考锚点。
        anchor_time: 参考锚点对应的事件时间。
        index: 目标元素在序列或缓冲区中的索引。
        observations: 观测消息序列。
        raw_points: 原始局部坐标点序列。
        config: 插件配置。

    Returns:
        Suspicion: 当前点的可疑性分类结果。
    """
    raw_point = raw_points[index]
    dt = max((observations[index].event_time - anchor_time).total_seconds(),
             config.min_dt_seconds)
    distance_to_anchor = distance(anchor_point, raw_point)
    implied_speed = distance_to_anchor / dt
    hard_jump = (implied_speed > config.prefilter_hard_speed_mps or
                 (dt <= config.prefilter_hard_distance_dt_seconds and
                  distance_to_anchor > config.prefilter_hard_distance_m))
    bridge_spike = (implied_speed > config.prefilter_soft_speed_mps and
                    is_bridge_spike_from_anchor(
                        observations=observations,
                        raw_points=raw_points,
                        anchor_point=anchor_point,
                        anchor_time=anchor_time,
                        index=index,
                        config=config,
                    ))
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
        observations: 观测消息序列。
        raw_points: 原始局部坐标点序列。
        anchor_point: 用于判断或修复的参考锚点。
        anchor_time: 参考锚点对应的事件时间。
        index: 目标元素在序列或缓冲区中的索引。
        config: 插件配置。

    Returns:
        bool: 当前点是否为基于锚点判断的桥接尖刺。
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
    if min(current_to_previous,
           current_to_next) <= config.prefilter_bridge_center_distance_m:
        return False

    left_dt = max(
        (observations[index].event_time - anchor_time).total_seconds(),
        config.min_dt_seconds)
    right_dt = dt_seconds(observations[index], observations[index + 1], config)
    return (left_dt <= config.prefilter_hard_distance_dt_seconds and
            right_dt <= config.prefilter_hard_distance_dt_seconds)
