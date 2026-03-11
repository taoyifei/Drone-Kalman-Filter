"""验收指标与基础统计的公共门面。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import parse_time
from drone_kalman_filter._metrics_acceptance import compute_device_acceptance
from drone_kalman_filter._metrics_alignment import (
    AcceptanceError,
    group_points_by_device as _group_points_by_device,
    group_points_by_track as _group_points_by_track,
    load_aligned_points as _load_aligned_points,
    split_segments_by_device as _split_segments_by_device,
)
from drone_kalman_filter._metrics_statistics import (
    as_float as _as_float,
    distribution as _distribution,
    haversine_m as _haversine_m,
    legacy_stats as _legacy_stats,
)


def build_report(path: str | Path) -> dict[str, Any]:
    """统计单个 JSONL 文件的基础运动指标。"""
    rows_by_track: dict[tuple[str, str, str | None],
                        list[dict[str, Any]]] = defaultdict(list)
    total_messages = 0

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            total_messages += 1
            identity = payload.get("identity") or {}
            source = payload.get("source") or {}
            spatial = payload.get("spatial") or {}
            position = spatial.get("position") or {}

            target_id = identity.get("targetId")
            device_id = source.get("deviceId")
            if not isinstance(target_id, str) or not isinstance(device_id, str):
                continue

            rows_by_track[(target_id, device_id,
                           identity.get("traceId"))].append({
                               "lat":
                                   _as_float(position.get("latitude")),
                               "lon":
                                   _as_float(position.get("longitude")),
                               "event_time":
                                   parse_time(payload.get("eventTime")),
                           })

    distances: list[float] = []
    speeds: list[float] = []
    point_pairs = 0

    for rows in rows_by_track.values():
        for previous, current in zip(rows, rows[1:]):
            if previous["lat"] is None or previous["lon"] is None or current[
                    "lat"] is None or current["lon"] is None:
                continue
            if previous["event_time"] is None or current["event_time"] is None:
                continue
            dt = (current["event_time"] -
                  previous["event_time"]).total_seconds()
            if dt <= 0:
                continue
            point_pairs += 1
            distance = _haversine_m(previous["lat"], previous["lon"],
                                    current["lat"], current["lon"])
            distances.append(distance)
            speeds.append(distance / dt)

    return {
        "messages": total_messages,
        "track_segments": len(rows_by_track),
        "point_pairs": point_pairs,
        "step_distance_m": _legacy_stats(distances),
        "implied_speed_mps": _legacy_stats(speeds),
    }


def compute_acceptance_metrics(
    raw_path: str | Path,
    smoothed_path: str | Path,
    config: PluginConfig,
) -> dict[str, Any]:
    """计算 raw 与 smoothed 的前端导向验收指标。"""
    points = _load_aligned_points(raw_path, smoothed_path)
    device_rows = _group_points_by_device(points)
    device_segments = _split_segments_by_device(_group_points_by_track(points),
                                                config)

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
                sorted(device_id for device_id, device in by_device.items()
                       if device["hard_gate_applies"]),
            "normal_point_offset_p95_m":
                _distribution(global_normal_offsets, include_max=True)["p95"],
            "global_offset_median_m":
                _distribution(global_offsets, include_max=True)["median"],
            "global_offset_p75_m":
                _distribution(global_offsets, include_max=True)["p75"],
            "global_offset_p95_m":
                _distribution(global_offsets, include_max=True)["p95"],
            "global_offset_p99_m":
                _distribution(global_offsets, include_max=True)["p99"],
            "recovery_points_p95":
                _distribution(global_recovery_points,
                              include_max=True,
                              treat_as_int=True)["p95"],
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
    """生成并写出验收摘要文件。"""
    summary = compute_acceptance_metrics(raw_path, smoothed_path, config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return summary


__all__ = [
    "AcceptanceError",
    "build_report",
    "compute_acceptance_metrics",
    "write_acceptance_summary",
]
