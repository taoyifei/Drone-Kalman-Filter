from __future__ import annotations

import json
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
class ComparePoint:
    line_no: int
    target_id: str | None
    device_id: str | None
    trace_id: str | None
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


def load_jsonl_lines(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def load_compare_points(
    raw_path: str | Path,
    smoothed_path: str | Path,
    *,
    error_type: type[ValueError],
) -> list[ComparePoint]:
    raw_lines = load_jsonl_lines(raw_path)
    smoothed_lines = load_jsonl_lines(smoothed_path)
    return build_compare_points(raw_lines, smoothed_lines, error_type=error_type)


def build_compare_points(
    raw_lines: Sequence[str],
    smoothed_lines: Sequence[str],
    *,
    error_type: type[ValueError],
) -> list[ComparePoint]:
    compare_points: list[ComparePoint] = []
    for line_no, (raw_line, smoothed_line) in enumerate(zip_longest(raw_lines, smoothed_lines), start=1):
        if raw_line is None or smoothed_line is None:
            raise error_type("Raw and smoothed files have different line counts.")
        raw_payload = _parse_json_line(raw_line, line_no, "raw", error_type)
        smoothed_payload = _parse_json_line(smoothed_line, line_no, "smoothed", error_type)
        compare_points.append(_build_compare_point(line_no, raw_payload, smoothed_payload, error_type))
    return compare_points


def group_points_by_device(points: Iterable[ComparePoint]) -> dict[tuple[str | None, str | None], list[ComparePoint]]:
    grouped: dict[tuple[str | None, str | None], list[ComparePoint]] = defaultdict(list)
    for point in points:
        grouped[(point.target_id, point.device_id)].append(point)
    return grouped


def split_track_segments(
    points: Sequence[ComparePoint],
    *,
    max_segment_gap_seconds: float,
) -> list[list[ComparePoint]]:
    segments: list[list[ComparePoint]] = []
    current: list[ComparePoint] = []

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


def _build_compare_point(
    line_no: int,
    raw_payload: dict[str, Any],
    smoothed_payload: dict[str, Any],
    error_type: type[ValueError],
) -> ComparePoint:
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
    return ComparePoint(
        line_no=line_no,
        target_id=raw_identity.get("targetId"),
        device_id=raw_source.get("deviceId"),
        trace_id=raw_identity.get("traceId"),
        event_time_text=raw_payload.get("eventTime"),
        event_time=event_time,
        raw_latitude=_as_float(raw_position.get("latitude")),
        raw_longitude=_as_float(raw_position.get("longitude")),
        smoothed_latitude=_as_float(smoothed_position.get("latitude")),
        smoothed_longitude=_as_float(smoothed_position.get("longitude")),
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
