from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Sequence

from drone_kalman_filter.message import parse_time


_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class PairedPoint:
    line_no: int
    target_id: str | None
    device_id: str | None
    trace_id: str | None
    object_type: int | None
    model: str | None
    event_time_text: str | None
    event_time: datetime
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


@dataclass(frozen=True, slots=True)
class TriplePoint:
    line_no: int
    target_id: str | None
    device_id: str | None
    trace_id: str | None
    event_time_text: str | None
    event_time: datetime
    raw_latitude: float | None
    raw_longitude: float | None
    current_latitude: float | None
    current_longitude: float | None
    baseline_latitude: float | None
    baseline_longitude: float | None

    @property
    def has_valid_coordinates(self) -> bool:
        return (
            self.raw_latitude is not None
            and self.raw_longitude is not None
            and self.current_latitude is not None
            and self.current_longitude is not None
            and self.baseline_latitude is not None
            and self.baseline_longitude is not None
        )


def load_jsonl_lines(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def load_paired_points(
    raw_path: str | Path,
    smoothed_path: str | Path,
    *,
    error_type: type[ValueError],
) -> list[PairedPoint]:
    raw_lines = load_jsonl_lines(raw_path)
    smoothed_lines = load_jsonl_lines(smoothed_path)
    return build_paired_points(raw_lines, smoothed_lines, error_type=error_type)


def build_paired_points(
    raw_lines: Sequence[str],
    smoothed_lines: Sequence[str],
    *,
    error_type: type[ValueError],
) -> list[PairedPoint]:
    paired_points: list[PairedPoint] = []
    for line_no, (raw_line, smoothed_line) in enumerate(zip_longest(raw_lines, smoothed_lines), start=1):
        if raw_line is None or smoothed_line is None:
            raise error_type("Raw and smoothed files have different line counts.")
        raw_payload = _parse_json_line(raw_line, line_no, "raw", error_type)
        smoothed_payload = _parse_json_line(smoothed_line, line_no, "smoothed", error_type)
        paired_points.append(_build_paired_point(line_no, raw_payload, smoothed_payload, error_type))
    return paired_points


def build_triple_points(
    raw_lines: Sequence[str],
    current_lines: Sequence[str],
    baseline_lines: Sequence[str],
    *,
    error_type: type[ValueError],
) -> list[TriplePoint]:
    triple_points: list[TriplePoint] = []
    for line_no, items in enumerate(zip_longest(raw_lines, current_lines, baseline_lines), start=1):
        raw_line, current_line, baseline_line = items
        if raw_line is None or current_line is None or baseline_line is None:
            raise error_type("Raw, current smoothed, and baseline files must have identical line counts.")
        raw_payload = _parse_json_line(raw_line, line_no, "raw", error_type)
        current_payload = _parse_json_line(current_line, line_no, "current", error_type)
        baseline_payload = _parse_json_line(baseline_line, line_no, "baseline", error_type)
        triple_points.append(
            _build_triple_point(
                line_no=line_no,
                raw_payload=raw_payload,
                current_payload=current_payload,
                baseline_payload=baseline_payload,
                error_type=error_type,
            )
        )
    return triple_points


def group_points_by_target(points: Iterable[PairedPoint]) -> dict[str | None, list[PairedPoint]]:
    grouped: dict[str | None, list[PairedPoint]] = defaultdict(list)
    for point in points:
        grouped[point.target_id].append(point)
    return grouped


def split_track_segments(
    points: Sequence[Any],
    *,
    max_segment_gap_seconds: float,
) -> list[list[Any]]:
    segments: list[list[Any]] = []
    current: list[Any] = []

    for point in points:
        if not point.has_valid_coordinates:
            if current:
                segments.append(current)
                current = []
            continue

        if not current:
            current = [point]
            continue

        previous = current[-1]
        delta = (point.event_time - previous.event_time).total_seconds()
        if (
            point.target_id != previous.target_id
            or point.device_id != previous.device_id
            or point.trace_id != previous.trace_id
            or delta <= 0.0
            or delta > max_segment_gap_seconds
        ):
            segments.append(current)
            current = [point]
            continue

        current.append(point)

    if current:
        segments.append(current)
    return segments


def reset_output_dir(output_dir: str | Path, *, subdirs: Sequence[str], files: Sequence[str]) -> None:
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    for subdir in subdirs:
        target = output_path / subdir
        if target.exists():
            shutil.rmtree(target)
    for filename in files:
        target = output_path / filename
        if target.exists():
            target.unlink()


def sanitize_filename(value: str | None, *, default: str) -> str:
    if not value:
        return default
    sanitized = _SAFE_FILENAME_PATTERN.sub("_", value).strip("._")
    return sanitized or default


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(value))


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_offset_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "max": None}
    return {
        "median": round(median(values), 4),
        "p95": round(percentile(values, 0.95), 4),
        "max": round(max(values), 4),
    }


def most_common_non_null(values: Iterable[Any]) -> Any:
    counts: dict[Any, int] = {}
    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _build_paired_point(
    line_no: int,
    raw_payload: dict[str, Any],
    smoothed_payload: dict[str, Any],
    error_type: type[ValueError],
) -> PairedPoint:
    raw_identity = raw_payload.get("identity") or {}
    raw_source = raw_payload.get("source") or {}
    smoothed_identity = smoothed_payload.get("identity") or {}
    smoothed_source = smoothed_payload.get("source") or {}

    raw_alignment = (
        raw_identity.get("targetId"),
        raw_source.get("deviceId"),
        raw_identity.get("traceId"),
        raw_payload.get("eventTime"),
    )
    smoothed_alignment = (
        smoothed_identity.get("targetId"),
        smoothed_source.get("deviceId"),
        smoothed_identity.get("traceId"),
        smoothed_payload.get("eventTime"),
    )
    if raw_alignment != smoothed_alignment:
        raise error_type(f"Alignment mismatch at line {line_no}: raw and smoothed metadata differ.")

    event_time = parse_time(raw_payload.get("eventTime"))
    if event_time is None:
        raise error_type(f"Invalid eventTime at line {line_no}.")

    raw_position = ((raw_payload.get("spatial") or {}).get("position") or {})
    smoothed_position = ((smoothed_payload.get("spatial") or {}).get("position") or {})
    return PairedPoint(
        line_no=line_no,
        target_id=raw_identity.get("targetId"),
        device_id=raw_source.get("deviceId"),
        trace_id=raw_identity.get("traceId"),
        object_type=raw_identity.get("type"),
        model=raw_identity.get("model"),
        event_time_text=raw_payload.get("eventTime"),
        event_time=event_time,
        raw_latitude=_as_float(raw_position.get("latitude")),
        raw_longitude=_as_float(raw_position.get("longitude")),
        smoothed_latitude=_as_float(smoothed_position.get("latitude")),
        smoothed_longitude=_as_float(smoothed_position.get("longitude")),
    )


def _build_triple_point(
    line_no: int,
    raw_payload: dict[str, Any],
    current_payload: dict[str, Any],
    baseline_payload: dict[str, Any],
    error_type: type[ValueError],
) -> TriplePoint:
    raw_identity = raw_payload.get("identity") or {}
    raw_source = raw_payload.get("source") or {}
    current_identity = current_payload.get("identity") or {}
    current_source = current_payload.get("source") or {}
    baseline_identity = baseline_payload.get("identity") or {}
    baseline_source = baseline_payload.get("source") or {}

    raw_alignment = (
        raw_identity.get("targetId"),
        raw_source.get("deviceId"),
        raw_identity.get("traceId"),
        raw_payload.get("eventTime"),
    )
    current_alignment = (
        current_identity.get("targetId"),
        current_source.get("deviceId"),
        current_identity.get("traceId"),
        current_payload.get("eventTime"),
    )
    baseline_alignment = (
        baseline_identity.get("targetId"),
        baseline_source.get("deviceId"),
        baseline_identity.get("traceId"),
        baseline_payload.get("eventTime"),
    )
    if raw_alignment != current_alignment or raw_alignment != baseline_alignment:
        raise error_type(f"Alignment mismatch at line {line_no} across raw/current/baseline.")

    event_time = parse_time(raw_payload.get("eventTime"))
    if event_time is None:
        raise error_type(f"Invalid eventTime at line {line_no}.")

    raw_position = ((raw_payload.get("spatial") or {}).get("position") or {})
    current_position = ((current_payload.get("spatial") or {}).get("position") or {})
    baseline_position = ((baseline_payload.get("spatial") or {}).get("position") or {})
    return TriplePoint(
        line_no=line_no,
        target_id=raw_identity.get("targetId"),
        device_id=raw_source.get("deviceId"),
        trace_id=raw_identity.get("traceId"),
        event_time_text=raw_payload.get("eventTime"),
        event_time=event_time,
        raw_latitude=_as_float(raw_position.get("latitude")),
        raw_longitude=_as_float(raw_position.get("longitude")),
        current_latitude=_as_float(current_position.get("latitude")),
        current_longitude=_as_float(current_position.get("longitude")),
        baseline_latitude=_as_float(baseline_position.get("latitude")),
        baseline_longitude=_as_float(baseline_position.get("longitude")),
    )


def _parse_json_line(
    line: str,
    line_no: int,
    label: str,
    error_type: type[ValueError],
) -> dict[str, Any]:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise error_type(f"Invalid JSON in {label} file at line {line_no}.") from exc


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
