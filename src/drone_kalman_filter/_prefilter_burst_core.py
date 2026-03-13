"""通用 burst 修复主路径。"""

from __future__ import annotations

from datetime import datetime

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import ParsedMessage
from drone_kalman_filter._prefilter_geometry import (
    distance,
    dt_seconds,
    interpolate_point,
)
from drone_kalman_filter._prefilter_types import (
    BurstAnchor,
    LocalPoint,
    TrustedAnchor,
)
from drone_kalman_filter._prefilter_suspicion import (
    _extend_suspicious_run,
    _initial_trusted_anchor,
    _qualifies_candidate_run,
    classify_suspicion,
    repair_single_point,
)


def repair_points_burst(
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    *,
    seed_anchor: TrustedAnchor | None = None,
) -> tuple[list[LocalPoint], bool, list[bool]]:
    """按 burst 规则修复连续异常点。

    Args:
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        tuple[list[LocalPoint], bool, list[bool]]:
            burst 路径下的修复结果、是否发生修复和改动标记。
    """
    (
        repaired,
        repaired_times,
        altered_flags,
        had_burst_repair,
        last_trusted_point,
        last_trusted_time,
        index,
    ) = _initial_burst_repair_state(
        observations=observations,
        raw_points=raw_points,
        seed_anchor=seed_anchor,
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
            (
                last_trusted_point,
                last_trusted_time,
                index,
            ) = _advance_trusted_point(
                repaired,
                repaired_times,
                altered_flags,
                observations,
                raw_points,
                index=index,
            )
            continue

        run_end = _extend_suspicious_run(
            start_index=index,
            anchor_point=last_trusted_point,
            anchor_time=last_trusted_time,
            observations=observations,
            raw_points=raw_points,
            config=config,
        )
        is_burst = _qualifies_candidate_run(
            anchor_point=last_trusted_point,
            raw_points=raw_points,
            start_index=index,
            run_end=run_end,
        )
        if not is_burst:
            (
                last_trusted_point,
                last_trusted_time,
                index,
            ) = _repair_non_burst_point(
                repaired,
                repaired_times,
                altered_flags,
                observations,
                raw_points,
                config=config,
                index=index,
                anchor_point=last_trusted_point,
                anchor_time=last_trusted_time,
            )
            continue

        (
            last_trusted_point,
            last_trusted_time,
            index,
            repaired_once,
        ) = _repair_burst_run(
            repaired,
            repaired_times,
            altered_flags,
            observations=observations,
            raw_points=raw_points,
            config=config,
            start_index=index,
            current_run_end=run_end,
            anchor_point=last_trusted_point,
            anchor_time=last_trusted_time,
        )
        had_burst_repair = had_burst_repair or repaired_once

    return repaired, had_burst_repair, altered_flags


def _initial_burst_repair_state(
    *,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    seed_anchor: TrustedAnchor | None,
) -> tuple[
    list[LocalPoint],
    list[datetime],
    list[bool],
    bool,
    LocalPoint,
    datetime,
    int,
]:
    """构建 burst 修复编排的初始状态。

    Args:
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        tuple[
            list[LocalPoint],
            list[datetime],
            list[bool],
            bool,
            LocalPoint,
            datetime,
            int,
        ]: repaired 列表、时间列表、改动标记、是否已修复、
        当前可信锚点、锚点时间和扫描起始索引。
    """
    repaired: list[LocalPoint] = []
    repaired_times: list[datetime] = []
    altered_flags: list[bool] = []
    had_burst_repair = False
    if seed_anchor is None:
        repaired.append(raw_points[0])
        repaired_times.append(observations[0].event_time)
        altered_flags.append(False)
    last_trusted_point, last_trusted_time, index = _initial_trusted_anchor(
        observations,
        raw_points,
        seed_anchor,
    )
    return (
        repaired,
        repaired_times,
        altered_flags,
        had_burst_repair,
        last_trusted_point,
        last_trusted_time,
        index,
    )


def _advance_trusted_point(
    repaired: list[LocalPoint],
    repaired_times: list[datetime],
    altered_flags: list[bool],
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    *,
    index: int,
) -> tuple[LocalPoint, datetime, int]:
    """追加一个可信原始点，并推进可信锚点。

    Args:
        repaired: 当前已修复的点序列。
        repaired_times: 当前已修复点的时间序列。
        altered_flags: 当前已修复点的改动标记序列。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        index: 当前扫描索引。

    Returns:
        tuple[LocalPoint, datetime, int]:
            新的可信锚点、锚点时间和下一次扫描索引。
    """
    _append_trusted_point(
        repaired,
        repaired_times,
        altered_flags,
        raw_points[index],
        observations[index].event_time,
    )
    return raw_points[index], observations[index].event_time, index + 1


def _repair_non_burst_point(
    repaired: list[LocalPoint],
    repaired_times: list[datetime],
    altered_flags: list[bool],
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    *,
    config: PluginConfig,
    index: int,
    anchor_point: LocalPoint,
    anchor_time: datetime,
) -> tuple[LocalPoint, datetime, int]:
    """修复不满足 burst 条件的单个可疑点。

    Args:
        repaired: 当前已修复的点序列。
        repaired_times: 当前已修复点的时间序列。
        altered_flags: 当前已修复点的改动标记序列。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        index: 当前扫描索引。
        anchor_point: 当前可信锚点。
        anchor_time: 当前可信锚点时间。

    Returns:
        tuple[LocalPoint, datetime, int]:
            修复后的点、修复时间和下一次扫描索引。
    """
    repaired_point = repair_single_point(
        index,
        observations,
        raw_points,
        repaired,
        repaired_times,
        anchor_point,
        anchor_time,
        config,
    )
    _append_repaired_point(
        repaired,
        repaired_times,
        altered_flags,
        repaired_point,
        observations[index].event_time,
    )
    return repaired_point, observations[index].event_time, index + 1


def _repair_burst_run(
    repaired: list[LocalPoint],
    repaired_times: list[datetime],
    altered_flags: list[bool],
    *,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    start_index: int,
    current_run_end: int,
    anchor_point: LocalPoint,
    anchor_time: datetime,
) -> tuple[LocalPoint, datetime, int, bool]:
    """修复一个满足条件的可疑 burst。

    Args:
        repaired: 当前已修复的点序列。
        repaired_times: 当前已修复点的时间序列。
        altered_flags: 当前已修复点的改动标记序列。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        start_index: burst 起点。
        current_run_end: 当前可疑 run 的结束索引。
        anchor_point: 当前可信锚点。
        anchor_time: 当前可信锚点时间。

    Returns:
        tuple[LocalPoint, datetime, int, bool]:
            新的可信锚点、锚点时间、下一次扫描索引以及是否发生修复。
    """
    anchor = find_future_anchor(
        start_index=start_index,
        current_run_end=current_run_end,
        anchor_point=anchor_point,
        anchor_time=anchor_time,
        observations=observations,
        raw_points=raw_points,
        config=config,
    )

    if anchor.anchor_index is None:
        _append_held_burst_points(
            repaired,
            repaired_times,
            altered_flags,
            observations,
            anchor_point=anchor_point,
            start_index=start_index,
            repair_end=anchor.repair_end,
        )
        return anchor_point, anchor_time, anchor.repair_end + 1, True

    _append_interpolated_burst_points(
        repaired,
        repaired_times,
        altered_flags,
        observations,
        raw_points,
        anchor_point=anchor_point,
        anchor_time=anchor_time,
        start_index=start_index,
        anchor_index=anchor.anchor_index,
    )
    repaired.append(raw_points[anchor.anchor_index])
    repaired_times.append(observations[anchor.anchor_index].event_time)
    altered_flags.append(False)
    return (
        raw_points[anchor.anchor_index],
        observations[anchor.anchor_index].event_time,
        anchor.anchor_index + 1,
        True,
    )


def _append_trusted_point(
    repaired: list[LocalPoint],
    repaired_times: list[datetime],
    altered_flags: list[bool],
    point: LocalPoint,
    event_time: datetime,
) -> None:
    """追加一个未修改的可信点。

    Args:
        repaired: 当前已修复的点序列。
        repaired_times: 当前已修复点的时间序列。
        altered_flags: 当前已修复点的改动标记序列。
        point: 待追加的点。
        event_time: 待追加点的时间。

    Returns:
        None: 不返回值。
    """
    repaired.append(point)
    repaired_times.append(event_time)
    altered_flags.append(False)


def _append_repaired_point(
    repaired: list[LocalPoint],
    repaired_times: list[datetime],
    altered_flags: list[bool],
    point: LocalPoint,
    event_time: datetime,
) -> None:
    """追加一个已修复的单点结果。

    Args:
        repaired: 当前已修复的点序列。
        repaired_times: 当前已修复点的时间序列。
        altered_flags: 当前已修复点的改动标记序列。
        point: 待追加的点。
        event_time: 待追加点的时间。

    Returns:
        None: 不返回值。
    """
    repaired.append(point)
    repaired_times.append(event_time)
    altered_flags.append(True)


def _append_held_burst_points(
    repaired: list[LocalPoint],
    repaired_times: list[datetime],
    altered_flags: list[bool],
    observations: list[ParsedMessage],
    *,
    anchor_point: LocalPoint,
    start_index: int,
    repair_end: int,
) -> None:
    """把无右锚点的 burst 全部保持在左锚点位置。

    Args:
        repaired: 当前已修复的点序列。
        repaired_times: 当前已修复点的时间序列。
        altered_flags: 当前已修复点的改动标记序列。
        observations: 当前窗口内的观测消息序列。
        anchor_point: 左锚点位置。
        start_index: 修复起点。
        repair_end: 修复终点。

    Returns:
        None: 不返回值。
    """
    for current_index in range(start_index, repair_end + 1):
        repaired.append(anchor_point)
        repaired_times.append(observations[current_index].event_time)
        altered_flags.append(True)


def _append_interpolated_burst_points(
    repaired: list[LocalPoint],
    repaired_times: list[datetime],
    altered_flags: list[bool],
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    *,
    anchor_point: LocalPoint,
    anchor_time: datetime,
    start_index: int,
    anchor_index: int,
) -> None:
    """按左右锚点时间插值，把 burst 区间写入输出序列。

    Args:
        repaired: 当前已修复的点序列。
        repaired_times: 当前已修复点的时间序列。
        altered_flags: 当前已修复点的改动标记序列。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        anchor_point: 左锚点位置。
        anchor_time: 左锚点时间。
        start_index: 修复起点。
        anchor_index: 右锚点索引。

    Returns:
        None: 不返回值。
    """
    for current_index in range(start_index, anchor_index):
        repaired.append(
            interpolate_point(
                left_point=anchor_point,
                right_point=raw_points[anchor_index],
                left_time=anchor_time,
                current_time=observations[current_index].event_time,
                right_time=observations[anchor_index].event_time,
            )
        )
        repaired_times.append(observations[current_index].event_time)
        altered_flags.append(True)


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
        anchor_point: 用于判定和修复的参考锚点。
        anchor_time: 参考锚点对应的事件时间。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。

    Returns:
        BurstAnchor: burst 修复找到的未来锚点信息。
    """
    repair_end = current_run_end
    search_limit = min(
        len(raw_points) - 1,
        start_index + config.prefilter_burst_max_run_length,
    )

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
            next_dt = dt_seconds(
                observations[candidate_index],
                observations[candidate_index + 1],
                config,
            )
            next_distance = distance(
                raw_points[candidate_index],
                raw_points[candidate_index + 1],
            )
            next_speed = next_distance / next_dt
            if next_speed > config.prefilter_hard_speed_mps or (
                next_dt <= config.prefilter_hard_distance_dt_seconds
                and next_distance > config.prefilter_hard_distance_m
            ):
                repair_end = candidate_index
                continue

        return BurstAnchor(
            repair_end=repair_end,
            anchor_index=candidate_index,
        )

    return BurstAnchor(repair_end=repair_end, anchor_index=None)
