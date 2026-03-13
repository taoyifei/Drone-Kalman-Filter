"""验收指标与基础统计的公共门面。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from drone_kalman_filter._metrics.acceptance import compute_device_acceptance
from drone_kalman_filter._metrics.alignment import (
    AcceptanceError,
    group_points_by_device as _group_points_by_device,
    group_points_by_track as _group_points_by_track,
    load_aligned_points as _load_aligned_points,
    split_segments_by_device as _split_segments_by_device,
)
from drone_kalman_filter._metrics.statistics import (
    as_float as _as_float,
    distribution as _distribution,
    haversine_m as _haversine_m,
    legacy_stats as _legacy_stats,
)
from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import parse_time


def build_report(path: str | Path) -> dict[str, Any]:
    """统计单个 JSONL 文件的基础运动指标。

    Args:
        path: 输入文件路径。

    Returns:
        dict[str, Any]: 轨迹统计报告。
    """
    total_messages, rows_by_track = _collect_track_rows(path)
    point_pairs, distances, speeds = _collect_track_statistics(rows_by_track)
    return {
        "messages": total_messages,
        "track_segments": len(rows_by_track),
        "point_pairs": point_pairs,
        "step_distance_m": _legacy_stats(distances),
        "implied_speed_mps": _legacy_stats(speeds),
    }


def _collect_track_rows(
    path: str | Path,
) -> tuple[
    int,
    dict[tuple[str, str, str | None], list[dict[str, Any]]],
]:
    """读取并按轨迹聚合 JSONL 中的坐标行。

    Args:
        path: 输入文件路径。

    Returns:
        tuple[int, dict[tuple[str, str, str | None], list[dict[str, Any]]]]:
            有效消息数和按轨迹分组的点列表。
    """
    rows_by_track: dict[
        tuple[str, str, str | None],
        list[dict[str, Any]],
    ] = defaultdict(list)
    total_messages = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            total_messages += 1
            track_key = _track_key(payload)
            if track_key is None:
                continue
            rows_by_track[track_key].append(_track_row(payload))
    return total_messages, rows_by_track


def _track_key(payload: dict[str, Any]) -> tuple[str, str, str | None] | None:
    """提取单条 payload 的轨迹键。

    Args:
        payload: 单条 JSON payload。

    Returns:
        tuple[str, str, str | None] | None:
            可用的轨迹键；若缺少必要标识则返回 None。
    """
    identity = payload.get("identity") or {}
    source = payload.get("source") or {}
    target_id = identity.get("targetId")
    device_id = source.get("deviceId")
    if not isinstance(target_id, str) or not isinstance(device_id, str):
        return None
    return target_id, device_id, identity.get("traceId")


def _track_row(payload: dict[str, Any]) -> dict[str, Any]:
    """从 payload 中提取用于轨迹统计的单行。

    Args:
        payload: 单条 JSON payload。

    Returns:
        dict[str, Any]: 坐标与时间都已解析的轨迹点数据。
    """
    spatial = payload.get("spatial") or {}
    position = spatial.get("position") or {}
    return {
        "lat": _as_float(position.get("latitude")),
        "lon": _as_float(position.get("longitude")),
        "event_time": parse_time(payload.get("eventTime")),
    }


def _collect_track_statistics(
    rows_by_track: dict[tuple[str, str, str | None], list[dict[str, Any]]],
) -> tuple[int, list[float], list[float]]:
    """统计所有轨迹的相邻点距离与速度。

    Args:
        rows_by_track: 按轨迹分组后的点。

    Returns:
        tuple[int, list[float], list[float]]:
            可用的点对数、步长列表和隐含速度列表。
    """
    distances: list[float] = []
    speeds: list[float] = []
    point_pairs = 0
    for rows in rows_by_track.values():
        point_pairs += _collect_row_pair_statistics(rows, distances, speeds)
    return point_pairs, distances, speeds


def _collect_row_pair_statistics(
    rows: list[dict[str, Any]],
    distances: list[float],
    speeds: list[float],
) -> int:
    """统计单条轨迹相邻点的距离和速度。

    Args:
        rows: 单条轨迹的点列表。
        distances: 需要追加的步长结果列表。
        speeds: 需要追加的隐含速度结果列表。

    Returns:
        int: 当前轨迹中可用的相邻点对数量。
    """
    point_pairs = 0
    for previous, current in zip(rows, rows[1:]):
        if _invalid_pair(previous, current):
            continue
        dt = (current["event_time"] - previous["event_time"]).total_seconds()
        if dt <= 0:
            continue
        point_pairs += 1
        distance = _haversine_m(
            previous["lat"],
            previous["lon"],
            current["lat"],
            current["lon"],
        )
        distances.append(distance)
        speeds.append(distance / dt)
    return point_pairs


def _invalid_pair(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """判断相邻点对是否缺少统计必要字段。

    Args:
        previous: 前一个轨迹点。
        current: 当前轨迹点。

    Returns:
        bool: 若缺少必要坐标或时间则返回 True。
    """
    missing_coordinates = (
        previous["lat"] is None
        or previous["lon"] is None
        or current["lat"] is None
        or current["lon"] is None
    )
    missing_time = (
        previous["event_time"] is None or current["event_time"] is None
    )
    return missing_coordinates or missing_time


def compute_acceptance_metrics(
    raw_path: str | Path,
    smoothed_path: str | Path,
    config: PluginConfig,
) -> dict[str, Any]:
    """计算 raw 与 smoothed 的前端导向验收指标。

    Args:
        raw_path: 原始 JSONL 文件路径。
        smoothed_path: 平滑后 JSONL 文件路径。
        config: 插件配置。

    Returns:
        dict[str, Any]: raw 与 smoothed 的验收指标。
    """
    points = _load_aligned_points(raw_path, smoothed_path)
    device_rows = _group_points_by_device(points)
    device_segments = _split_segments_by_device(
        _group_points_by_track(points),
        config,
    )

    by_device: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, Any]] = []
    global_offsets: list[float] = []
    global_normal_offsets: list[float] = []
    global_recovery_points: list[int] = []
    global_raw_flip_count = 0
    global_smoothed_flip_count = 0

    for device_id in sorted(device_rows):
        result = compute_device_acceptance(
            device_id=device_id,
            rows=device_rows[device_id],
            segments=device_segments.get(device_id, []),
            config=config,
        )
        by_device[device_id] = result.summary
        violations.extend(result.violations)
        global_offsets.extend(result.global_offset_values)
        global_normal_offsets.extend(result.normal_offset_values)
        global_recovery_points.extend(result.recovery_points)
        global_raw_flip_count += result.raw_flip_count
        global_smoothed_flip_count += result.smoothed_flip_count

    return {
        "global": {
            "total_points":
                sum(device["point_count"] for device in by_device.values()),
            "device_count":
                len(by_device),
            "hard_gate_devices":
                sorted(
                    device_id
                    for device_id, device in by_device.items()
                    if device["hard_gate_applies"]
                ),
            "normal_point_offset_p95_m":
                _distribution(
                    global_normal_offsets,
                    include_max=True,
                )["p95"],
            "global_offset_median_m":
                _distribution(global_offsets, include_max=True)["median"],
            "global_offset_p75_m":
                _distribution(global_offsets, include_max=True)["p75"],
            "global_offset_p95_m":
                _distribution(global_offsets, include_max=True)["p95"],
            "global_offset_p99_m":
                _distribution(global_offsets, include_max=True)["p99"],
            "recovery_points_p95":
                _distribution(
                    global_recovery_points,
                    include_max=True,
                    treat_as_int=True,
                )["p95"],
            "raw_direction_flip_count":
                global_raw_flip_count,
            "direction_flip_count":
                global_smoothed_flip_count,
        },
        "by_device": by_device,
        "violations": violations,
    }


def write_acceptance_summary(
    raw_path: str | Path,
    smoothed_path: str | Path,
    output_path: str | Path,
    config: PluginConfig,
) -> dict[str, Any]:
    """生成并写出验收摘要文件。

    Args:
        raw_path: 原始 JSONL 文件路径。
        smoothed_path: 平滑后 JSONL 文件路径。
        output_path: 输出文件路径。
        config: 插件配置。

    Returns:
        dict[str, Any]: 已写入磁盘的验收摘要。
    """
    summary = compute_acceptance_metrics(raw_path, smoothed_path, config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "AcceptanceError",
    "build_report",
    "compute_acceptance_metrics",
    "write_acceptance_summary",
]
