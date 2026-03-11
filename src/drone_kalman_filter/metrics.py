from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .config import PluginConfig
from .message import parse_time


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
class _AlignedPoint:
    # raw / smoothed 按行对齐后，用统一结构做验收指标计算。
    line_no: int
    target_id: str | None
    device_id: str | None
    trace_id: str | None
    event_time_text: str | None
    event_time: object
    raw_latitude: float | None
    raw_longitude: float | None
    smoothed_latitude: float | None
    smoothed_longitude: float | None

    @property
    def has_valid_coordinates(self) -> bool:
        return (
            self.raw_latitude is not None
            and self.raw_longitude is not None
            and self.smoothed_latitude is not None
            and self.smoothed_longitude is not None
        )


class AcceptanceError(ValueError):
    pass


def build_report(path: str | Path) -> dict[str, Any]:
    # report 是粗粒度统计，只回答“整体跳变有没有下降”。
    rows_by_track: dict[tuple[str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
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

            key = (target_id, device_id, identity.get("traceId"))
            latitude = _as_float(position.get("latitude"))
            longitude = _as_float(position.get("longitude"))
            event_time = parse_time(payload.get("eventTime"))
            rows_by_track[key].append({"lat": latitude, "lon": longitude, "event_time": event_time})

    distances: list[float] = []
    speeds: list[float] = []
    point_pairs = 0

    for rows in rows_by_track.values():
        for previous, current in zip(rows, rows[1:]):
            if previous["lat"] is None or previous["lon"] is None or current["lat"] is None or current["lon"] is None:
                continue
            if previous["event_time"] is None or current["event_time"] is None:
                continue
            dt = (current["event_time"] - previous["event_time"]).total_seconds()
            if dt <= 0:
                continue
            point_pairs += 1
            distance = _haversine_m(previous["lat"], previous["lon"], current["lat"], current["lon"])
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
    # acceptance 是更贴近前端目标的验收：看偏移、恢复速度和折返次数。
    points = _load_aligned_points(raw_path, smoothed_path)
    device_rows = _group_points_by_device(points)
    track_rows = _group_points_by_track(points)
    device_segments = _split_segments_by_device(track_rows, config)

    by_device: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, Any]] = []
    global_offsets: list[float] = []
    global_normal_offsets: list[float] = []
    global_recovery_points: list[int] = []
    global_raw_flip_count = 0
    global_smoothed_flip_count = 0

    for device_id in sorted(device_rows):
        # 验收按 device 分开算，因为前端也是按 deviceId 分别画线。
        rows = device_rows[device_id]
        segments = device_segments.get(device_id, [])
        dt_values = _device_dt_values(rows)
        latency_values = _estimate_release_latencies(rows, config)

        global_offset_values: list[float] = []
        normal_offset_values: list[float] = []
        recovery_points: list[int] = []
        raw_flip_count = 0
        smoothed_flip_count = 0
        normal_offset_violations: list[dict[str, Any]] = []
        recovery_violations: list[dict[str, Any]] = []
        latency_violations: list[dict[str, Any]] = []

        for segment in segments:
            offsets = [_offset_m(point) for point in segment]
            jump_tags = _jump_exclusion_mask(segment, config)
            global_offset_values.extend(offsets)
            global_offsets.extend(offsets)

            for index, offset in enumerate(offsets):
                if jump_tags[index]:
                    continue
                normal_offset_values.append(offset)
                global_normal_offsets.append(offset)
                normal_offset_violations.append(
                    {
                        "eventTime": segment[index].event_time_text,
                        "deviceId": device_id,
                        "metric": "normal_point_offset_p95_m",
                        "value": round(offset, 4),
                        "line_no": segment[index].line_no,
                    }
                )

            segment_recoveries = _recovery_measurements(segment, offsets, config)
            recovery_points.extend(value["recovery_points"] for value in segment_recoveries)
            global_recovery_points.extend(value["recovery_points"] for value in segment_recoveries)
            recovery_violations.extend(
                {
                    "eventTime": value["eventTime"],
                    "deviceId": device_id,
                    "metric": "recovery_points_p95",
                    "value": value["recovery_points"],
                    "line_no": value["line_no"],
                }
                for value in segment_recoveries
            )

        raw_flip_count = _count_direction_flips(rows, use_smoothed=False)
        smoothed_flip_count = _count_direction_flips(rows, use_smoothed=True)

        global_raw_flip_count += raw_flip_count
        global_smoothed_flip_count += smoothed_flip_count

        point_count = sum(len(segment) for segment in segments)
        dt_stats = _distribution(dt_values, include_max=True)
        latency_stats = _distribution(latency_values, include_max=True)
        normal_offset_stats = _distribution(normal_offset_values, include_max=True)
        global_offset_stats = _distribution(global_offset_values, include_max=True)
        recovery_stats = _distribution(recovery_points, include_max=True, treat_as_int=True)
        is_dense = (dt_stats["p95"] or 0.0) <= DENSE_DEVICE_DT_P95_THRESHOLD_S if dt_stats["p95"] is not None else False
        warning = (
            "direction_flip_count exceeds raw_direction_flip_count on this device"
            if smoothed_flip_count > raw_flip_count
            else None
        )

        by_device[device_id] = {
            "point_count": point_count,
            "dense_device": is_dense,
            "hard_gate_applies": point_count >= MIN_POINTS_FOR_HARD_GATE,
            "dt_median_s": dt_stats["median"],
            "dt_p95_s": dt_stats["p95"],
            "normal_point_count": len(normal_offset_values),
            "normal_point_offset_p95_m": normal_offset_stats["p95"],
            "global_offset_median_m": global_offset_stats["median"],
            "global_offset_p75_m": global_offset_stats["p75"],
            "global_offset_p95_m": global_offset_stats["p95"],
            "global_offset_p99_m": global_offset_stats["p99"],
            "recovery_points_p95": recovery_stats["p95"],
            "raw_direction_flip_count": raw_flip_count,
            "direction_flip_count": smoothed_flip_count,
            "release_latency_median_s": latency_stats["median"],
            "release_latency_p95_s": latency_stats["p95"],
            "warning": warning,
        }

        if point_count >= MIN_POINTS_FOR_HARD_GATE and normal_offset_stats["p95"] is not None:
            if normal_offset_stats["p95"] > NORMAL_POINT_OFFSET_THRESHOLD_M:
                violations.extend(
                    sorted(normal_offset_violations, key=lambda item: item["value"], reverse=True)[:5]
                )

        if recovery_stats["p95"] is not None and recovery_stats["p95"] > RECOVERY_POINTS_THRESHOLD:
            violations.extend(sorted(recovery_violations, key=lambda item: item["value"], reverse=True)[:5])

        if is_dense:
            dense_limit = _dense_latency_limit(dt_stats["p95"], config)
            if latency_stats["median"] is not None and latency_stats["median"] > 3.0:
                latency_violations.extend(
                    {
                        "eventTime": rows[index].event_time_text,
                        "deviceId": device_id,
                        "metric": "release_latency_median_s",
                        "value": round(value, 4),
                        "line_no": rows[index].line_no,
                    }
                    for index, value in enumerate(latency_values)
                )
            if latency_stats["p95"] is not None and latency_stats["p95"] > dense_limit:
                latency_violations.extend(
                    {
                        "eventTime": rows[index].event_time_text,
                        "deviceId": device_id,
                        "metric": "release_latency_p95_s",
                        "value": round(value, 4),
                        "line_no": rows[index].line_no,
                    }
                    for index, value in enumerate(latency_values)
                )
        else:
            if latency_stats["max"] is not None and latency_stats["max"] > config.idle_flush_seconds:
                latency_violations.extend(
                    {
                        "eventTime": rows[index].event_time_text,
                        "deviceId": device_id,
                        "metric": "release_latency_idle_flush_s",
                        "value": round(value, 4),
                        "line_no": rows[index].line_no,
                    }
                    for index, value in enumerate(latency_values)
                )

        if latency_violations:
            metric_name = latency_violations[0]["metric"]
            violations.extend(
                sorted(
                    [item for item in latency_violations if item["metric"] == metric_name],
                    key=lambda item: item["value"],
                    reverse=True,
                )[:5]
            )

    summary = {
        "global": {
            "total_points": sum(device["point_count"] for device in by_device.values()),
            "device_count": len(by_device),
            "hard_gate_devices": sorted(
                device_id for device_id, device in by_device.items() if device["hard_gate_applies"]
            ),
            "normal_point_offset_p95_m": _distribution(global_normal_offsets, include_max=True)["p95"],
            "global_offset_median_m": _distribution(global_offsets, include_max=True)["median"],
            "global_offset_p75_m": _distribution(global_offsets, include_max=True)["p75"],
            "global_offset_p95_m": _distribution(global_offsets, include_max=True)["p95"],
            "global_offset_p99_m": _distribution(global_offsets, include_max=True)["p99"],
            "recovery_points_p95": _distribution(global_recovery_points, include_max=True, treat_as_int=True)["p95"],
            "raw_direction_flip_count": global_raw_flip_count,
            "direction_flip_count": global_smoothed_flip_count,
        },
        "by_device": by_device,
        "violations": violations,
    }
    return summary


def write_acceptance_summary(
    raw_path: str | Path,
    smoothed_path: str | Path,
    output_path: str | Path,
    config: PluginConfig,
) -> dict[str, Any]:
    summary = compute_acceptance_metrics(raw_path, smoothed_path, config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _load_aligned_points(raw_path: str | Path, smoothed_path: str | Path) -> list[_AlignedPoint]:
    # raw 和 smoothed 必须逐行严格对齐，否则所有验收指标都没有意义。
    raw_lines = Path(raw_path).read_text(encoding="utf-8").splitlines()
    smoothed_lines = Path(smoothed_path).read_text(encoding="utf-8").splitlines()
    if len(raw_lines) != len(smoothed_lines):
        raise AcceptanceError("Raw and smoothed files have different line counts.")

    points: list[_AlignedPoint] = []
    for line_no, (raw_line, smoothed_line) in enumerate(zip(raw_lines, smoothed_lines), start=1):
        raw_payload = json.loads(raw_line)
        smoothed_payload = json.loads(smoothed_line)
        raw_identity = raw_payload.get("identity") or {}
        raw_source = raw_payload.get("source") or {}
        smoothed_identity = smoothed_payload.get("identity") or {}
        smoothed_source = smoothed_payload.get("source") or {}

        raw_key = (
            raw_identity.get("targetId"),
            raw_source.get("deviceId"),
            raw_identity.get("traceId"),
            raw_payload.get("eventTime"),
        )
        smoothed_key = (
            smoothed_identity.get("targetId"),
            smoothed_source.get("deviceId"),
            smoothed_identity.get("traceId"),
            smoothed_payload.get("eventTime"),
        )
        if raw_key != smoothed_key:
            raise AcceptanceError(f"Alignment mismatch at line {line_no}.")

        raw_position = ((raw_payload.get("spatial") or {}).get("position") or {})
        smoothed_position = ((smoothed_payload.get("spatial") or {}).get("position") or {})
        event_time = parse_time(raw_payload.get("eventTime"))
        if event_time is None:
            raise AcceptanceError(f"Invalid eventTime at line {line_no}.")

        points.append(
            _AlignedPoint(
                line_no=line_no,
                target_id=_as_str(raw_identity.get("targetId")),
                device_id=_as_str(raw_source.get("deviceId")),
                trace_id=_as_str(raw_identity.get("traceId")),
                event_time_text=raw_payload.get("eventTime"),
                event_time=event_time,
                raw_latitude=_as_float(raw_position.get("latitude")),
                raw_longitude=_as_float(raw_position.get("longitude")),
                smoothed_latitude=_as_float(smoothed_position.get("latitude")),
                smoothed_longitude=_as_float(smoothed_position.get("longitude")),
            )
        )

    return points


def _group_points_by_device(points: list[_AlignedPoint]) -> dict[str, list[_AlignedPoint]]:
    grouped: dict[str, list[_AlignedPoint]] = defaultdict(list)
    for point in points:
        if point.device_id is None or not point.has_valid_coordinates:
            continue
        grouped[point.device_id].append(point)
    for device_points in grouped.values():
        device_points.sort(key=lambda item: item.line_no)
    return grouped


def _group_points_by_track(points: list[_AlignedPoint]) -> dict[tuple[str, str], list[_AlignedPoint]]:
    grouped: dict[tuple[str, str], list[_AlignedPoint]] = defaultdict(list)
    for point in points:
        if point.device_id is None or point.target_id is None or not point.has_valid_coordinates:
            continue
        grouped[(point.target_id, point.device_id)].append(point)
    for track_points in grouped.values():
        track_points.sort(key=lambda item: item.line_no)
    return grouped


def _split_segments_by_device(
    tracks: dict[tuple[str, str], list[_AlignedPoint]],
    config: PluginConfig,
) -> dict[str, list[list[_AlignedPoint]]]:
    # 这里的切段规则要和实时主链路一致，否则验收会和实际表现脱节。
    by_device: dict[str, list[list[_AlignedPoint]]] = defaultdict(list)
    for (_, device_id), points in tracks.items():
        current: list[_AlignedPoint] = []
        for point in points:
            if not current:
                current = [point]
                continue
            previous = current[-1]
            delta = (point.event_time - previous.event_time).total_seconds()
            if (
                point.trace_id != previous.trace_id
                or delta <= 0.0
                or delta > config.max_segment_gap_seconds
            ):
                by_device[device_id].append(current)
                current = [point]
                continue
            current.append(point)
        if current:
            by_device[device_id].append(current)
    return by_device


def _device_dt_values(points: list[_AlignedPoint]) -> list[float]:
    values: list[float] = []
    for previous, current in zip(points, points[1:]):
        delta = (current.event_time - previous.event_time).total_seconds()
        if delta > 0.0:
            values.append(delta)
    return values


def _estimate_release_latencies(points: list[_AlignedPoint], config: PluginConfig) -> list[float]:
    # 这里是离线估算延迟，不依赖运行时埋点。
    values: list[float] = []
    for index, point in enumerate(points):
        candidate_index = index + config.lag_points
        if candidate_index < len(points):
            delta = (points[candidate_index].event_time - point.event_time).total_seconds()
            values.append(min(max(delta, 0.0), config.idle_flush_seconds))
        else:
            values.append(config.idle_flush_seconds)
    return values


def _jump_exclusion_mask(segment: list[_AlignedPoint], config: PluginConfig) -> list[bool]:
    # 高疑似跳变点及其后续少量点不计入“正常点偏移”。
    excluded = [False] * len(segment)
    for index in range(1, len(segment)):
        speed = _raw_implied_speed(segment[index - 1], segment[index], config)
        if speed <= JUMP_TAG_SPEED_MPS:
            continue
        stop = min(len(segment), index + NORMAL_POINT_EXCLUSION_AFTER + 1)
        for tagged_index in range(index, stop):
            excluded[tagged_index] = True
    return excluded


def _recovery_measurements(
    segment: list[_AlignedPoint],
    offsets: list[float],
    config: PluginConfig,
) -> list[dict[str, Any]]:
    # 只对明显大跳变统计“多久回到正常轨迹附近”。
    measurements: list[dict[str, Any]] = []
    for index in range(1, len(segment)):
        speed = _raw_implied_speed(segment[index - 1], segment[index], config)
        if speed <= RECOVERY_JUMP_SPEED_MPS:
            continue
        recovery_points = len(segment) - index
        for candidate_index in range(index + 1, len(segment)):
            if offsets[candidate_index] < RECOVERY_OFFSET_THRESHOLD_M:
                recovery_points = candidate_index - index
                break
        measurements.append(
            {
                "line_no": segment[index].line_no,
                "eventTime": segment[index].event_time_text,
                "recovery_points": recovery_points,
            }
        )
    return measurements


def _count_direction_flips(segment: list[_AlignedPoint], *, use_smoothed: bool) -> int:
    return len(_direction_flip_events(segment, use_smoothed=use_smoothed))


def _direction_flip_events(segment: list[_AlignedPoint], *, use_smoothed: bool) -> list[float]:
    # 用三点转向角统计高频折返，越多通常说明前端看起来越抖。
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

        ab = _vector_m(left_xy[0], left_xy[1], center_xy[0], center_xy[1])
        bc = _vector_m(center_xy[0], center_xy[1], right_xy[0], right_xy[1])
        ab_norm = math.hypot(*ab)
        bc_norm = math.hypot(*bc)
        if ab_norm <= DIRECTION_FLIP_MIN_STEP_M or bc_norm <= DIRECTION_FLIP_MIN_STEP_M:
            continue

        angle_deg = _vector_angle_deg(ab, bc)
        if angle_deg > DIRECTION_FLIP_ANGLE_DEG:
            events.append(angle_deg)
    return events


def _vector_m(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    east = _haversine_m(lat1, lon1, lat1, lon2)
    north = _haversine_m(lat1, lon1, lat2, lon1)
    if lon2 < lon1:
        east *= -1.0
    if lat2 < lat1:
        north *= -1.0
    return east, north


def _vector_angle_deg(left: tuple[float, float], right: tuple[float, float]) -> float:
    numerator = left[0] * right[0] + left[1] * right[1]
    denominator = math.hypot(*left) * math.hypot(*right)
    if denominator == 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, numerator / denominator))
    return math.degrees(math.acos(cosine))


def _offset_m(point: _AlignedPoint) -> float:
    return _haversine_m(
        point.raw_latitude,
        point.raw_longitude,
        point.smoothed_latitude,
        point.smoothed_longitude,
    )


def _raw_implied_speed(previous: _AlignedPoint, current: _AlignedPoint, config: PluginConfig) -> float:
    distance = _haversine_m(
        previous.raw_latitude,
        previous.raw_longitude,
        current.raw_latitude,
        current.raw_longitude,
    )
    delta = (current.event_time - previous.event_time).total_seconds()
    dt_eff = max(delta, config.min_dt_seconds)
    return distance / dt_eff


def _dense_latency_limit(dt_p95: float | None, config: PluginConfig) -> float:
    if dt_p95 is None:
        return float(config.idle_flush_seconds)
    return config.lag_points * dt_p95


def _distribution(
    values: list[float] | list[int],
    *,
    include_max: bool,
    treat_as_int: bool = False,
) -> dict[str, float | int | None]:
    if not values:
        base = {"count": 0, "median": None, "p75": None, "p95": None, "p99": None}
        if include_max:
            base["max"] = None
        return base

    ordered = sorted(float(value) for value in values)
    result: dict[str, float | int | None] = {
        "count": len(ordered),
        "median": _round_stat(_percentile(ordered, 0.50), treat_as_int=treat_as_int),
        "p75": _round_stat(_percentile(ordered, 0.75), treat_as_int=treat_as_int),
        "p95": _round_stat(_percentile(ordered, 0.95), treat_as_int=treat_as_int),
        "p99": _round_stat(_percentile(ordered, 0.99), treat_as_int=treat_as_int),
    }
    if include_max:
        result["max"] = _round_stat(max(ordered), treat_as_int=treat_as_int)
    return result


def _legacy_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p95": None, "p99": None, "max": None}

    return {
        "count": len(values),
        "median": round(median(values), 4),
        "p95": round(_percentile(values, 0.95), 4),
        "p99": round(_percentile(values, 0.99), 4),
        "max": round(max(values), 4),
    }


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _round_stat(value: float, *, treat_as_int: bool) -> float | int:
    if treat_as_int:
        return int(math.ceil(value))
    return round(value, 4)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(a))
