"""fusion 设备专用的微突跳修复逻辑。"""

from __future__ import annotations

from datetime import datetime

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import ParsedMessage
from drone_kalman_filter._prefilter_geometry import (
    distance,
    dt_seconds,
    interpolate_point,
)
from drone_kalman_filter._prefilter_types import LocalPoint, TrustedAnchor

FUSION_DEVICE_ID = "fusion"
FUSION_JUMP_SPEED_MPS = 60.0
FUSION_JUMP_DISTANCE_M = 40.0
FUSION_STABLE_SPEED_MPS = 25.0
FUSION_STABLE_DISTANCE_M = 20.0
FUSION_MIN_BURST_STEPS = 2
FUSION_REQUIRED_STABLE_STEPS = 2


def repair_points_fusion_micro_burst(
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    *,
    seed_anchor: TrustedAnchor | None = None,
) -> tuple[list[LocalPoint], list[bool]] | None:
    """修复 fusion 设备上的短时高频微突跳。

    Args:
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        tuple[list[LocalPoint], list[bool]] | None:
            命中 fusion 微突跳路径时，返回修复后的点序列与改动标记；
            否则返回 None。
    """
    if not _supports_fusion_micro_burst(observations, raw_points):
        return None

    repaired, altered_flags, index = _initial_fusion_repair_state(
        raw_points,
        seed_anchor,
    )
    had_repair = False
    while index < len(raw_points):
        index, repaired_once, is_done = _repair_next_fusion_step(
            repaired,
            altered_flags,
            observations=observations,
            raw_points=raw_points,
            config=config,
            seed_anchor=seed_anchor,
            start_index=index,
        )
        had_repair = had_repair or repaired_once
        if is_done:
            break

    if not had_repair:
        return None
    return repaired, altered_flags


def _supports_fusion_micro_burst(
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
) -> bool:
    """判断当前窗口是否可以进入 fusion 微突跳修复路径。

    Args:
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。

    Returns:
        bool: 仅当数据来自 fusion 且点数满足最小要求时返回 True。
    """
    return (
        _is_fusion_device(observations)
        and len(raw_points) >= FUSION_MIN_BURST_STEPS + 2
    )


def _repair_next_fusion_step(
    repaired: list[LocalPoint],
    altered_flags: list[bool],
    *,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    seed_anchor: TrustedAnchor | None,
    start_index: int,
) -> tuple[int, bool, bool]:
    """修复下一个 fusion 微突跳候选，或直接追加后续原始点。

    Args:
        repaired: 当前已修复的点序列。
        altered_flags: 当前已修复点的改动标记序列。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续下来的可信锚点。
        start_index: 当前扫描起点。

    Returns:
        tuple[int, bool, bool]:
            下一次扫描的起始索引、这一步是否发生了修复、以及是否可
            以直接结束整个扫描。
    """
    burst_start = _find_fusion_burst_start(
        start_index=start_index,
        observations=observations,
        raw_points=raw_points,
        config=config,
        seed_anchor=seed_anchor,
    )
    if burst_start is None:
        _append_raw_points(
            repaired,
            altered_flags,
            raw_points,
            start=start_index,
            stop=len(raw_points),
        )
        return len(raw_points), False, True

    _append_fusion_gap(
        repaired,
        altered_flags,
        raw_points,
        index=start_index,
        burst_start=burst_start,
    )
    next_index, repaired_once = _repair_fusion_burst(
        repaired,
        altered_flags,
        observations,
        raw_points,
        config,
        seed_anchor,
        burst_start,
    )
    return next_index, repaired_once, False


def _initial_fusion_repair_state(
    raw_points: list[LocalPoint],
    seed_anchor: TrustedAnchor | None,
) -> tuple[list[LocalPoint], list[bool], int]:
    """初始化 fusion 微突跳修复输出状态。

    Args:
        raw_points: 当前窗口内的局部平面原始点序列。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        tuple[list[LocalPoint], list[bool], int]:
            初始 repaired 列表、改动标记和扫描起点索引。
    """
    repaired: list[LocalPoint] = []
    altered_flags: list[bool] = []
    index = 0
    if seed_anchor is None:
        repaired.append(raw_points[0])
        altered_flags.append(False)
        index = 1
    return repaired, altered_flags, index


def _append_fusion_gap(
    repaired: list[LocalPoint],
    altered_flags: list[bool],
    raw_points: list[LocalPoint],
    *,
    index: int,
    burst_start: int,
) -> None:
    """补上 fusion burst 前未改动的原始点。

    Args:
        repaired: 当前已修复的点序列。
        altered_flags: 当前已修复点的改动标记序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        index: 当前扫描位置。
        burst_start: burst 起点。

    Returns:
        None: 不返回值。
    """
    if burst_start <= index:
        return
    _append_raw_points(
        repaired,
        altered_flags,
        raw_points,
        start=index,
        stop=burst_start,
    )


def _repair_fusion_burst(
    repaired: list[LocalPoint],
    altered_flags: list[bool],
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    seed_anchor: TrustedAnchor | None,
    burst_start: int,
) -> tuple[int, bool]:
    """修复一次 fusion 微突跳 burst。

    Args:
        repaired: 当前已修复的点序列。
        altered_flags: 当前已修复点的改动标记序列。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续下来的可信锚点。
        burst_start: burst 起点。

    Returns:
        tuple[int, bool]: 下一次扫描的起始索引，以及本次是否发生修复。
    """
    burst_end = _find_fusion_burst_end(
        start_index=burst_start,
        observations=observations,
        raw_points=raw_points,
        config=config,
        seed_anchor=seed_anchor,
    )
    left_anchor_point, left_anchor_time = _fusion_left_anchor(
        burst_start=burst_start,
        observations=observations,
        raw_points=raw_points,
        seed_anchor=seed_anchor,
    )
    right_anchor_index = _find_fusion_stable_anchor(
        burst_end=burst_end,
        observations=observations,
        raw_points=raw_points,
        config=config,
        seed_anchor=seed_anchor,
    )
    if right_anchor_index is None:
        for _ in range(burst_start, burst_end + 1):
            repaired.append(left_anchor_point)
            altered_flags.append(True)
        return burst_end + 1, True
    _append_interpolated_fusion_points(
        repaired,
        altered_flags,
        observations,
        raw_points,
        burst_start=burst_start,
        right_anchor_index=right_anchor_index,
        left_anchor_point=left_anchor_point,
        left_anchor_time=left_anchor_time,
    )
    return right_anchor_index + 1, True


def _append_interpolated_fusion_points(
    repaired: list[LocalPoint],
    altered_flags: list[bool],
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    *,
    burst_start: int,
    right_anchor_index: int,
    left_anchor_point: LocalPoint,
    left_anchor_time: datetime,
) -> None:
    """按时间插值把 fusion burst 区间回写到输出序列。

    Args:
        repaired: 当前已修复的点序列。
        altered_flags: 当前已修复点的改动标记序列。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        burst_start: burst 起点。
        right_anchor_index: 右锚点索引。
        left_anchor_point: 左锚点位置。
        left_anchor_time: 左锚点时间。

    Returns:
        None: 不返回值。
    """
    right_anchor_point = raw_points[right_anchor_index]
    right_anchor_time = observations[right_anchor_index].event_time
    for burst_index in range(burst_start, right_anchor_index):
        repaired.append(
            interpolate_point(
                left_point=left_anchor_point,
                right_point=right_anchor_point,
                left_time=left_anchor_time,
                current_time=observations[burst_index].event_time,
                right_time=right_anchor_time,
            )
        )
        altered_flags.append(True)
    repaired.append(right_anchor_point)
    altered_flags.append(False)


def _is_fusion_device(observations: list[ParsedMessage]) -> bool:
    """判断当前窗口是否来自 fusion 设备。

    Args:
        observations: 当前窗口内的观测消息序列。

    Returns:
        bool: 当前窗口是否来自 fusion 设备。
    """
    if not observations:
        return False
    track_key = observations[0].track_key
    return track_key is not None and track_key.device_id == FUSION_DEVICE_ID


def _append_raw_points(
    repaired: list[LocalPoint],
    altered_flags: list[bool],
    raw_points: list[LocalPoint],
    *,
    start: int,
    stop: int,
) -> None:
    """把一段未修改的原始点写入输出序列。

    Args:
        repaired: 当前已修复的点序列。
        altered_flags: 当前已修复点的改动标记序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        start: 起始索引。
        stop: 结束索引，采用开区间。

    Returns:
        None: 不返回值。
    """
    for point in raw_points[start:stop]:
        repaired.append(point)
        altered_flags.append(False)


def _find_fusion_burst_start(
    *,
    start_index: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    seed_anchor: TrustedAnchor | None,
) -> int | None:
    """查找当前窗口里第一个 fusion burst 起点。

    Args:
        start_index: 当前扫描起点。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        int | None: 命中 burst 起点时返回其索引，否则返回 None。
    """
    search_stop = len(raw_points) - FUSION_MIN_BURST_STEPS + 1
    for index in range(start_index, search_stop):
        if not _fusion_step_is_jump(
            point_index=index,
            observations=observations,
            raw_points=raw_points,
            config=config,
            seed_anchor=seed_anchor,
        ):
            continue
        if _fusion_step_is_jump(
            point_index=index + 1,
            observations=observations,
            raw_points=raw_points,
            config=config,
            seed_anchor=seed_anchor,
        ):
            return index
    return None


def _find_fusion_burst_end(
    *,
    start_index: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    seed_anchor: TrustedAnchor | None,
) -> int:
    """扩展 fusion burst，直到相邻异常步停止。

    Args:
        start_index: burst 起点。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        int: burst 终点索引。
    """
    burst_end = start_index + FUSION_MIN_BURST_STEPS - 1
    while burst_end + 1 < len(raw_points):
        if not _fusion_step_is_jump(
            point_index=burst_end + 1,
            observations=observations,
            raw_points=raw_points,
            config=config,
            seed_anchor=seed_anchor,
        ):
            break
        burst_end += 1
    return burst_end


def _find_fusion_stable_anchor(
    *,
    burst_end: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    seed_anchor: TrustedAnchor | None,
) -> int | None:
    """查找开始连续稳定恢复的未来锚点。

    Args:
        burst_end: burst 终点。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        int | None: 右锚点索引；当前窗口内不可见时返回 None。
    """
    last_candidate = len(raw_points) - FUSION_REQUIRED_STABLE_STEPS
    for candidate_index in range(burst_end + 1, last_candidate + 1):
        is_stable = True
        for offset in range(FUSION_REQUIRED_STABLE_STEPS):
            if not _fusion_step_is_stable(
                point_index=candidate_index + offset,
                observations=observations,
                raw_points=raw_points,
                config=config,
                seed_anchor=seed_anchor,
            ):
                is_stable = False
                break
        if is_stable:
            return candidate_index
    return None


def _fusion_left_anchor(
    *,
    burst_start: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    seed_anchor: TrustedAnchor | None,
) -> tuple[LocalPoint, datetime]:
    """返回 fusion burst 左侧最后一个稳定锚点。

    Args:
        burst_start: burst 起点。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        tuple[LocalPoint, datetime]: 左锚点及其时间。
    """
    if burst_start == 0:
        if seed_anchor is None:
            return raw_points[0], observations[0].event_time
        return seed_anchor.point, seed_anchor.event_time
    left_index = burst_start - 1
    return raw_points[left_index], observations[left_index].event_time


def _fusion_step_is_jump(
    *,
    point_index: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    seed_anchor: TrustedAnchor | None,
) -> bool:
    """判断当前相邻步是否属于 fusion 微突跳。

    Args:
        point_index: 当前步终点索引。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        bool: 当前相邻步是否满足 fusion 微突跳条件。
    """
    previous_point, previous_time = _fusion_step_previous(
        point_index=point_index,
        observations=observations,
        raw_points=raw_points,
        seed_anchor=seed_anchor,
    )
    if previous_point is None or previous_time is None:
        return False
    step_distance = distance(previous_point, raw_points[point_index])
    step_dt = max(
        (observations[point_index].event_time - previous_time)
        .total_seconds(),
        config.min_dt_seconds,
    )
    step_speed = step_distance / step_dt
    return (
        step_speed > FUSION_JUMP_SPEED_MPS
        or step_distance > FUSION_JUMP_DISTANCE_M
    )


def _fusion_step_is_stable(
    *,
    point_index: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    seed_anchor: TrustedAnchor | None,
) -> bool:
    """判断当前相邻步是否足够稳定，可作为恢复锚点。

    Args:
        point_index: 当前步终点索引。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        config: 插件配置。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        bool: 当前相邻步是否满足稳定恢复条件。
    """
    previous_point, previous_time = _fusion_step_previous(
        point_index=point_index,
        observations=observations,
        raw_points=raw_points,
        seed_anchor=seed_anchor,
    )
    if previous_point is None or previous_time is None:
        return False
    step_distance = distance(previous_point, raw_points[point_index])
    step_dt = max(
        (observations[point_index].event_time - previous_time)
        .total_seconds(),
        config.min_dt_seconds,
    )
    step_speed = step_distance / step_dt
    return (
        step_speed <= FUSION_STABLE_SPEED_MPS
        and step_distance <= FUSION_STABLE_DISTANCE_M
    )


def _fusion_step_previous(
    *,
    point_index: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    seed_anchor: TrustedAnchor | None,
) -> tuple[LocalPoint | None, datetime | None]:
    """解析当前相邻步所使用的前一个点。

    Args:
        point_index: 当前步终点索引。
        observations: 当前窗口内的观测消息序列。
        raw_points: 当前窗口内的局部平面原始点序列。
        seed_anchor: 跨窗口延续下来的可信锚点。

    Returns:
        tuple[LocalPoint | None, datetime | None]:
            当前步的前一个点及其时间；无法判定时返回 None。
    """
    if point_index <= 0:
        if seed_anchor is None:
            return None, None
        return seed_anchor.point, seed_anchor.event_time
    return raw_points[point_index - 1], observations[point_index - 1].event_time
