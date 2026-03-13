"""设备级验收指标与违规点聚合。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter._metrics.alignment import AlignedPoint
from drone_kalman_filter._metrics.statistics import (
    distribution,
    haversine_m,
    vector_angle_deg,
    vector_m,
)

JUMP_TAG_SPEED_MPS = 45.0
RECOVERY_JUMP_SPEED_MPS = 100.0
NORMAL_POINT_EXCLUSION_AFTER = 5
NORMAL_POINT_OFFSET_THRESHOLD_M = 50.0
MIN_POINTS_FOR_HARD_GATE = 200
DENSE_DEVICE_DT_P95_THRESHOLD_S = 10.0
DIRECTION_FLIP_MIN_STEP_M = 10.0
DIRECTION_FLIP_ANGLE_DEG = 120.0
RECOVERY_OFFSET_THRESHOLD_M = 20.0
RECOVERY_POINTS_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class DeviceAcceptanceResult:
    """保存单个 device 的验收结果和全局聚合材料。"""

    summary: dict[str, object]
    violations: list[dict[str, object]]
    global_offset_values: list[float]
    normal_offset_values: list[float]
    recovery_points: list[int]
    raw_flip_count: int
    smoothed_flip_count: int


@dataclass(frozen=True, slots=True)
class _SegmentMeasurements:
    """保存按片段收集出的中间测量结果。"""

    global_offset_values: list[float]
    normal_offset_values: list[float]
    recovery_points: list[int]
    normal_offset_violations: list[dict[str, object]]
    recovery_violations: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class _DeviceStats:
    """保存单设备统计分布和派生标记。"""

    point_count: int
    dt_values: list[float]
    latency_values: list[float]
    dt_stats: dict[str, float | int | None]
    latency_stats: dict[str, float | int | None]
    normal_offset_stats: dict[str, float | int | None]
    global_offset_stats: dict[str, float | int | None]
    recovery_stats: dict[str, float | int | None]
    is_dense: bool


def compute_device_acceptance(
    device_id: str,
    rows: list[AlignedPoint],
    segments: list[list[AlignedPoint]],
    config: PluginConfig,
) -> DeviceAcceptanceResult:
    """计算单个 device 的验收指标与违规点。

    Args:
        device_id: 设备标识。
        rows: 当前设备对应的对齐点序列。
        segments: 按轨迹切分后的片段序列。
        config: 插件配置。

    Returns:
        DeviceAcceptanceResult: 单设备的验收结果。
    """
    measurements = _collect_segment_measurements(device_id, segments, config)
    raw_flip_count, smoothed_flip_count = _compute_flip_totals(segments)
    stats = _compute_device_stats(
        segments,
        config,
        measurements.global_offset_values,
        measurements.normal_offset_values,
        measurements.recovery_points,
    )
    latency_violations = _build_latency_violations(
        device_id,
        rows,
        stats.latency_values,
        stats.latency_stats,
        stats.dt_stats,
        stats.is_dense,
        config,
    )
    return _build_device_result(
        measurements,
        latency_violations,
        stats,
        raw_flip_count,
        smoothed_flip_count,
    )


def _collect_segment_measurements(
    device_id: str,
    segments: list[list[AlignedPoint]],
    config: PluginConfig,
) -> _SegmentMeasurements:
    """按片段收集 offset、恢复事件和违规候选。

    Args:
        device_id: 设备标识。
        segments: 当前设备的连续片段列表。
        config: 插件配置。

    Returns:
        _SegmentMeasurements: 片段级聚合得到的中间结果。
    """
    global_offset_values: list[float] = []
    normal_offset_values: list[float] = []
    recovery_points: list[int] = []
    normal_offset_violations: list[dict[str, object]] = []
    recovery_violations: list[dict[str, object]] = []

    for segment in segments:
        offsets = [offset_m(point) for point in segment]
        jump_tags = jump_exclusion_mask(segment, config)
        global_offset_values.extend(offsets)
        normal_offset_values.extend(
            offset
            for offset, excluded in zip(offsets, jump_tags)
            if not excluded
        )
        normal_offset_violations.extend(
            _build_normal_offset_violation(device_id, point, offset)
            for point, offset, excluded in zip(segment, offsets, jump_tags)
            if not excluded
        )

        segment_recoveries = recovery_measurements(segment, offsets, config)
        recovery_points.extend(
            value["recovery_points"]
            for value in segment_recoveries
        )
        recovery_violations.extend(
            _build_recovery_violation(device_id, value)
            for value in segment_recoveries
        )

    return _SegmentMeasurements(
        global_offset_values=global_offset_values,
        normal_offset_values=normal_offset_values,
        recovery_points=recovery_points,
        normal_offset_violations=normal_offset_violations,
        recovery_violations=recovery_violations,
    )


def _build_normal_offset_violation(
    device_id: str,
    point: AlignedPoint,
    offset: float,
) -> dict[str, object]:
    """构造单个正常点偏移违规候选。

    Args:
        device_id: 设备标识。
        point: 当前点。
        offset: 当前点的 raw/smoothed 偏移距离。

    Returns:
        dict[str, object]: 违规候选记录。
    """
    return {
        "eventTime": point.event_time_text,
        "deviceId": device_id,
        "metric": "normal_point_offset_p95_m",
        "value": round(offset, 4),
        "line_no": point.line_no,
    }


def _build_recovery_violation(
    device_id: str,
    value: dict[str, object],
) -> dict[str, object]:
    """构造单个恢复事件违规候选。

    Args:
        device_id: 设备标识。
        value: 恢复事件测量记录。

    Returns:
        dict[str, object]: 违规候选记录。
    """
    return {
        "eventTime": value["eventTime"],
        "deviceId": device_id,
        "metric": "recovery_points_p95",
        "value": value["recovery_points"],
        "line_no": value["line_no"],
    }


def _compute_flip_totals(
    segments: list[list[AlignedPoint]],
) -> tuple[int, int]:
    """汇总所有片段的 raw/smoothed 折返次数。

    Args:
        segments: 当前设备的连续片段列表。

    Returns:
        tuple[int, int]: raw 折返总数和 smoothed 折返总数。
    """
    raw_flip_count = sum(
        count_direction_flips(segment, use_smoothed=False)
        for segment in segments
    )
    smoothed_flip_count = sum(
        count_direction_flips(segment, use_smoothed=True)
        for segment in segments
    )
    return raw_flip_count, smoothed_flip_count


def _compute_device_stats(
    segments: list[list[AlignedPoint]],
    config: PluginConfig,
    global_offset_values: list[float],
    normal_offset_values: list[float],
    recovery_points: list[int],
) -> _DeviceStats:
    """计算单设备的分布统计与密度标记。

    Args:
        segments: 当前设备的连续片段列表。
        config: 插件配置。
        global_offset_values: 全量点偏移列表。
        normal_offset_values: 正常点偏移列表。
        recovery_points: 跳变恢复点数列表。

    Returns:
        _DeviceStats: 聚合后的统计结果。
    """
    dt_values = flatten_dt_values(segments)
    latency_values = flatten_latency_values(segments, config)
    point_count = sum(len(segment) for segment in segments)
    dt_stats = distribution(dt_values, include_max=True)
    latency_stats = distribution(latency_values, include_max=True)
    normal_offset_stats = distribution(normal_offset_values, include_max=True)
    global_offset_stats = distribution(global_offset_values, include_max=True)
    recovery_stats = distribution(
        recovery_points,
        include_max=True,
        treat_as_int=True,
    )
    is_dense = (
        (dt_stats["p95"] or 0.0) <= DENSE_DEVICE_DT_P95_THRESHOLD_S
        if dt_stats["p95"] is not None else False
    )
    return _DeviceStats(
        point_count=point_count,
        dt_values=dt_values,
        latency_values=latency_values,
        dt_stats=dt_stats,
        latency_stats=latency_stats,
        normal_offset_stats=normal_offset_stats,
        global_offset_stats=global_offset_stats,
        recovery_stats=recovery_stats,
        is_dense=is_dense,
    )


def _build_latency_violations(
    device_id: str,
    rows: list[AlignedPoint],
    latency_values: list[float],
    latency_stats: dict[str, float | int | None],
    dt_stats: dict[str, float | int | None],
    is_dense: bool,
    config: PluginConfig,
) -> list[dict[str, object]]:
    """按 dense/sparse 规则构造延迟违规候选。

    Args:
        device_id: 设备标识。
        rows: 当前设备对应的对齐点序列。
        latency_values: 估计输出延迟列表。
        latency_stats: 延迟分布统计。
        dt_stats: 时间间隔分布统计。
        is_dense: 当前设备是否视为稠密设备。
        config: 插件配置。

    Returns:
        list[dict[str, object]]: 延迟违规候选列表。
    """
    latency_violations: list[dict[str, object]] = []

    if is_dense:
        dense_limit = dense_latency_limit(dt_stats["p95"], config)
        if (
            latency_stats["median"] is not None
            and latency_stats["median"] > 3.0
        ):
            latency_violations.extend(
                _build_latency_violation(
                    rows[index],
                    device_id,
                    "release_latency_median_s",
                    value,
                )
                for index, value in enumerate(latency_values)
            )
        if (
            latency_stats["p95"] is not None
            and latency_stats["p95"] > dense_limit
        ):
            latency_violations.extend(
                _build_latency_violation(
                    rows[index],
                    device_id,
                    "release_latency_p95_s",
                    value,
                )
                for index, value in enumerate(latency_values)
            )
    else:
        if (
            latency_stats["max"] is not None
            and latency_stats["max"] > config.idle_flush_seconds
        ):
            latency_violations.extend(
                _build_latency_violation(
                    rows[index],
                    device_id,
                    "release_latency_idle_flush_s",
                    value,
                )
                for index, value in enumerate(latency_values)
            )

    return latency_violations


def _build_latency_violation(
    point: AlignedPoint,
    device_id: str,
    metric: str,
    value: float,
) -> dict[str, object]:
    """构造单个延迟违规候选。

    Args:
        point: 当前点。
        device_id: 设备标识。
        metric: 指标名。
        value: 指标值。

    Returns:
        dict[str, object]: 延迟违规候选记录。
    """
    return {
        "eventTime": point.event_time_text,
        "deviceId": device_id,
        "metric": metric,
        "value": round(value, 4),
        "line_no": point.line_no,
    }


def _build_device_result(
    measurements: _SegmentMeasurements,
    latency_violations: list[dict[str, object]],
    stats: _DeviceStats,
    raw_flip_count: int,
    smoothed_flip_count: int,
) -> DeviceAcceptanceResult:
    """组装单设备 summary 和 top 违规列表。

    Args:
        measurements: 片段级中间测量结果。
        latency_violations: 延迟违规候选列表。
        stats: 单设备聚合统计。
        raw_flip_count: raw 折返总数。
        smoothed_flip_count: smoothed 折返总数。

    Returns:
        DeviceAcceptanceResult: 组装完成的单设备结果。
    """
    warning = (
        "direction_flip_count exceeds raw_direction_flip_count on this device"
        if smoothed_flip_count > raw_flip_count else None
    )
    top_normal_offset_violations = _top_normal_offset_violations(
        measurements.normal_offset_violations,
        stats.point_count,
        stats.normal_offset_stats,
        latency_violations,
    )
    top_recovery_violations = _top_recovery_violations(
        measurements.recovery_violations,
        stats.recovery_stats,
    )
    top_latency_violations = _top_latency_violations(latency_violations)
    summary = {
        "point_count": stats.point_count,
        "dense_device": stats.is_dense,
        "hard_gate_applies": stats.point_count >= MIN_POINTS_FOR_HARD_GATE,
        "dt_median_s": stats.dt_stats["median"],
        "dt_p95_s": stats.dt_stats["p95"],
        "normal_point_count": len(measurements.normal_offset_values),
        "normal_point_offset_p95_m": stats.normal_offset_stats["p95"],
        "global_offset_median_m": stats.global_offset_stats["median"],
        "global_offset_p75_m": stats.global_offset_stats["p75"],
        "global_offset_p95_m": stats.global_offset_stats["p95"],
        "global_offset_p99_m": stats.global_offset_stats["p99"],
        "recovery_points_p95": stats.recovery_stats["p95"],
        "raw_direction_flip_count": raw_flip_count,
        "direction_flip_count": smoothed_flip_count,
        "release_latency_median_s": stats.latency_stats["median"],
        "release_latency_p95_s": stats.latency_stats["p95"],
        "warning": warning,
    }
    return DeviceAcceptanceResult(
        summary=summary,
        violations=(
            top_normal_offset_violations
            + top_recovery_violations
            + top_latency_violations
        ),
        global_offset_values=measurements.global_offset_values,
        normal_offset_values=measurements.normal_offset_values,
        recovery_points=measurements.recovery_points,
        raw_flip_count=raw_flip_count,
        smoothed_flip_count=smoothed_flip_count,
    )


def _top_normal_offset_violations(
    normal_offset_violations: list[dict[str, object]],
    point_count: int,
    normal_offset_stats: dict[str, float | int | None],
    latency_violations: list[dict[str, object]],
) -> list[dict[str, object]]:
    """按当前门槛裁剪正常点偏移违规列表。

    Args:
        normal_offset_violations: 正常点偏移违规候选。
        point_count: 当前设备总点数。
        normal_offset_stats: 正常点偏移统计。
        latency_violations: 延迟违规列表，会保留当前空操作语义。

    Returns:
        list[dict[str, object]]: 保留下来的 top 违规点。
    """
    if (
        point_count >= MIN_POINTS_FOR_HARD_GATE
        and normal_offset_stats["p95"] is not None
    ):
        if normal_offset_stats["p95"] > NORMAL_POINT_OFFSET_THRESHOLD_M:
            latency_violations.extend([])
            return sorted(
                normal_offset_violations,
                key=lambda item: item["value"],
                reverse=True,
            )[:5]
    return []


def _top_recovery_violations(
    recovery_violations: list[dict[str, object]],
    recovery_stats: dict[str, float | int | None],
) -> list[dict[str, object]]:
    """按当前门槛裁剪恢复违规列表。

    Args:
        recovery_violations: 恢复违规候选。
        recovery_stats: 恢复点数统计。

    Returns:
        list[dict[str, object]]: 保留下来的 top 违规点。
    """
    if (
        recovery_stats["p95"] is not None
        and recovery_stats["p95"] > RECOVERY_POINTS_THRESHOLD
    ):
        return sorted(
            recovery_violations,
            key=lambda item: item["value"],
            reverse=True,
        )[:5]
    return []


def _top_latency_violations(
    latency_violations: list[dict[str, object]],
) -> list[dict[str, object]]:
    """按首个 metric 分组裁剪延迟违规列表。

    Args:
        latency_violations: 延迟违规候选。

    Returns:
        list[dict[str, object]]: 保留下来的 top 违规点。
    """
    if not latency_violations:
        return []
    metric_name = latency_violations[0]["metric"]
    return sorted(
        [
            item
            for item in latency_violations
            if item["metric"] == metric_name
        ],
        key=lambda item: item["value"],
        reverse=True,
    )[:5]


def estimate_release_dt_values(points: list[AlignedPoint]) -> list[float]:
    """统计同一设备相邻点之间的时间间隔。

    Args:
        points: 点序列。

    Returns:
        list[float]: 释放输出之间的时间间隔列表。
    """
    values: list[float] = []
    for previous, current in zip(points, points[1:]):
        delta = (current.event_time - previous.event_time).total_seconds()
        if delta > 0.0:
            values.append(delta)
    return values


def flatten_dt_values(segments: list[list[AlignedPoint]]) -> list[float]:
    """汇总所有片段内部的相邻时间间隔。

    Args:
        segments: 当前设备的连续片段列表。

    Returns:
        list[float]: 所有片段内部的时间间隔集合。
    """
    values: list[float] = []
    for segment in segments:
        values.extend(estimate_release_dt_values(segment))
    return values


def estimate_release_latencies(points: list[AlignedPoint],
                               config: PluginConfig) -> list[float]:
    """离线估算 fixed-lag 输出延迟。

    Args:
        points: 点序列。
        config: 插件配置。

    Returns:
        list[float]: 估计得到的输出延迟列表。
    """
    values: list[float] = []
    for index, point in enumerate(points):
        candidate_index = index + config.lag_points
        if candidate_index < len(points):
            delta = (points[candidate_index].event_time -
                     point.event_time).total_seconds()
            values.append(min(max(delta, 0.0), config.idle_flush_seconds))
        else:
            values.append(config.idle_flush_seconds)
    return values


def flatten_latency_values(
    segments: list[list[AlignedPoint]],
    config: PluginConfig,
) -> list[float]:
    """汇总所有片段的估计输出延迟。

    Args:
        segments: 当前设备的连续片段列表。
        config: 插件配置。

    Returns:
        list[float]: 所有片段的 fixed-lag 延迟估计。
    """
    values: list[float] = []
    for segment in segments:
        values.extend(estimate_release_latencies(segment, config))
    return values


def jump_exclusion_mask(segment: list[AlignedPoint],
                        config: PluginConfig) -> list[bool]:
    """标记需要排除出正常点统计的跳变区间。

    Args:
        segment: 单个轨迹片段。
        config: 插件配置。

    Returns:
        list[bool]: 需要排除的跳变点布尔掩码。
    """
    excluded = [False] * len(segment)
    for index in range(1, len(segment)):
        speed = raw_implied_speed(segment[index - 1], segment[index], config)
        if speed <= JUMP_TAG_SPEED_MPS:
            continue
        stop = min(len(segment), index + NORMAL_POINT_EXCLUSION_AFTER + 1)
        for tagged_index in range(index, stop):
            excluded[tagged_index] = True
    return excluded


def recovery_measurements(
    segment: list[AlignedPoint],
    offsets: list[float],
    config: PluginConfig,
) -> list[dict[str, object]]:
    """统计明显大跳变恢复到正常范围所需的点数。

    Args:
        segment: 单个轨迹片段。
        offsets: 偏移量序列。
        config: 插件配置。

    Returns:
        list[dict[str, object]]: 恢复阶段的诊断测量记录。
    """
    measurements: list[dict[str, object]] = []
    for index in range(1, len(segment)):
        speed = raw_implied_speed(segment[index - 1], segment[index], config)
        if speed <= RECOVERY_JUMP_SPEED_MPS:
            continue
        recovery_points = len(segment) - index
        for candidate_index in range(index + 1, len(segment)):
            if offsets[candidate_index] < RECOVERY_OFFSET_THRESHOLD_M:
                recovery_points = candidate_index - index
                break
        measurements.append({
            "line_no": segment[index].line_no,
            "eventTime": segment[index].event_time_text,
            "recovery_points": recovery_points,
        })
    return measurements


def count_direction_flips(segment: list[AlignedPoint], *,
                          use_smoothed: bool) -> int:
    """统计一段轨迹中的高频折返次数。

    Args:
        segment: 单个轨迹片段。
        use_smoothed: 是否使用平滑后坐标。

    Returns:
        int: 方向翻转次数。
    """
    return len(direction_flip_events(segment, use_smoothed=use_smoothed))


def direction_flip_events(segment: list[AlignedPoint], *,
                          use_smoothed: bool) -> list[float]:
    """找出一段轨迹中所有满足条件的折返事件。

    Args:
        segment: 单个轨迹片段。
        use_smoothed: 是否使用平滑后坐标。

    Returns:
        list[float]: 发生方向翻转时的时间点列表。
    """
    events: list[float] = []
    for left, center, right in zip(segment, segment[1:], segment[2:]):
        if use_smoothed:
            left_xy = (left.smoothed_latitude, left.smoothed_longitude)
            center_xy = (center.smoothed_latitude, center.smoothed_longitude)
            right_xy = (right.smoothed_latitude, right.smoothed_longitude)
        else:
            left_xy = (left.raw_latitude, left.raw_longitude)
            center_xy = (center.raw_latitude, center.raw_longitude)
            right_xy = (right.raw_latitude, right.raw_longitude)

        if None in left_xy or None in center_xy or None in right_xy:
            continue

        ab = vector_m(left_xy[0], left_xy[1], center_xy[0], center_xy[1])
        bc = vector_m(center_xy[0], center_xy[1], right_xy[0], right_xy[1])
        ab_norm = math.hypot(*ab)
        bc_norm = math.hypot(*bc)
        if (ab_norm <= DIRECTION_FLIP_MIN_STEP_M or
                bc_norm <= DIRECTION_FLIP_MIN_STEP_M):
            continue

        angle_deg = vector_angle_deg(ab, bc)
        if angle_deg > DIRECTION_FLIP_ANGLE_DEG:
            events.append(angle_deg)
    return events


def offset_m(point: AlignedPoint) -> float:
    """计算单个点 raw 与 smoothed 之间的距离。

    Args:
        point: 单个点。

    Returns:
        float: 点相对基准的位移距离，单位为米。
    """
    return haversine_m(
        point.raw_latitude,
        point.raw_longitude,
        point.smoothed_latitude,
        point.smoothed_longitude,
    )


def raw_implied_speed(previous: AlignedPoint, current: AlignedPoint,
                      config: PluginConfig) -> float:
    """按原始相邻点估算隐含速度。

    Args:
        previous: 前一个观测或时间点。
        current: 当前观测或时间点。
        config: 插件配置。

    Returns:
        float: 原始点对推导出的速度，单位为米每秒。
    """
    distance = haversine_m(
        previous.raw_latitude,
        previous.raw_longitude,
        current.raw_latitude,
        current.raw_longitude,
    )
    delta = (current.event_time - previous.event_time).total_seconds()
    dt_eff = max(delta, config.min_dt_seconds)
    return distance / dt_eff


def dense_latency_limit(dt_p95: float | None, config: PluginConfig) -> float:
    """给稠密设备计算允许的延迟上限。

    Args:
        dt_p95: 输入参数。
        config: 插件配置。

    Returns:
        float: 稠密采样情况下允许的延迟上限。
    """
    if dt_p95 is None:
        return float(config.idle_flush_seconds)
    return config.lag_points * dt_p95
