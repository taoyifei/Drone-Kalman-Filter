"""raw 与 smoothed 的逐行对齐和按设备切段。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import parse_time
from drone_kalman_filter._metrics_statistics import as_float, as_str


@dataclass(frozen=True, slots=True)
class AlignedPoint:
    """保存按行对齐后的 raw 与 smoothed 点。"""

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
        """判断对齐后的 raw 和 smoothed 是否都有可用坐标。

        Args:
            None. 不接收额外参数。

        Returns:
            bool: 坐标字段是否完整且可用。
        """
        return (self.raw_latitude is not None and
                self.raw_longitude is not None and
                self.smoothed_latitude is not None and
                self.smoothed_longitude is not None)


class AcceptanceError(ValueError):
    """表示 raw 与 smoothed 验收输入无法对齐。"""


def load_aligned_points(raw_path: str | Path,
                        smoothed_path: str | Path) -> list[AlignedPoint]:
    """逐行读取并对齐 raw 与 smoothed 数据。

    Args:
        raw_path: 原始 JSONL 文件路径。
        smoothed_path: 平滑后 JSONL 文件路径。

    Returns:
        list[AlignedPoint]: 按行对齐后的点列表。

    Raises:
        AcceptanceError: 当 raw 与 smoothed 文件无法逐行对齐时抛出。
    """
    raw_lines = Path(raw_path).read_text(encoding="utf-8").splitlines()
    smoothed_lines = Path(smoothed_path).read_text(
        encoding="utf-8").splitlines()
    if len(raw_lines) != len(smoothed_lines):
        raise AcceptanceError(
            "Raw and smoothed files have different line counts.")

    points: list[AlignedPoint] = []
    for line_no, (raw_line,
                  smoothed_line) in enumerate(zip(raw_lines, smoothed_lines),
                                              start=1):
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

        raw_position = ((raw_payload.get("spatial") or {}).get("position") or
                        {})
        smoothed_position = ((smoothed_payload.get("spatial") or
                              {}).get("position") or {})
        event_time = parse_time(raw_payload.get("eventTime"))
        if event_time is None:
            raise AcceptanceError(f"Invalid eventTime at line {line_no}.")

        points.append(
            AlignedPoint(
                line_no=line_no,
                target_id=as_str(raw_identity.get("targetId")),
                device_id=as_str(raw_source.get("deviceId")),
                trace_id=as_str(raw_identity.get("traceId")),
                event_time_text=raw_payload.get("eventTime"),
                event_time=event_time,
                raw_latitude=as_float(raw_position.get("latitude")),
                raw_longitude=as_float(raw_position.get("longitude")),
                smoothed_latitude=as_float(smoothed_position.get("latitude")),
                smoothed_longitude=as_float(smoothed_position.get("longitude")),
            ))

    return points


def group_points_by_device(
        points: list[AlignedPoint]) -> dict[str, list[AlignedPoint]]:
    """按 deviceId 分组有效点。

    Args:
        points: 点序列。

    Returns:
        dict[str, list[AlignedPoint]]: 按设备分组后的点集合。
    """
    grouped: dict[str, list[AlignedPoint]] = defaultdict(list)
    for point in points:
        if point.device_id is None or not point.has_valid_coordinates:
            continue
        grouped[point.device_id].append(point)
    for device_points in grouped.values():
        device_points.sort(key=lambda item: item.line_no)
    return grouped


def group_points_by_track(
        points: list[AlignedPoint]
) -> dict[tuple[str, str], list[AlignedPoint]]:
    """按 targetId 和 deviceId 分组有效点。

    Args:
        points: 点序列。

    Returns:
        dict[tuple[str, str], list[AlignedPoint]]: 按轨迹分组后的点集合。
    """
    grouped: dict[tuple[str, str], list[AlignedPoint]] = defaultdict(list)
    for point in points:
        if (point.device_id is None or point.target_id is None or
                not point.has_valid_coordinates):
            continue
        grouped[(point.target_id, point.device_id)].append(point)
    for track_points in grouped.values():
        track_points.sort(key=lambda item: item.line_no)
    return grouped


def split_segments_by_device(
    tracks: dict[tuple[str, str], list[AlignedPoint]],
    config: PluginConfig,
) -> dict[str, list[list[AlignedPoint]]]:
    """按实时主链路同样的规则把轨迹切成连续段。

    Args:
        tracks: 按轨迹分组后的点集合。
        config: 插件配置。

    Returns:
        dict[str, list[list[AlignedPoint]]]: 按设备和间断切分后的片段。
    """
    by_device: dict[str, list[list[AlignedPoint]]] = defaultdict(list)
    for (_, device_id), points in tracks.items():
        current: list[AlignedPoint] = []
        for point in points:
            if not current:
                current = [point]
                continue
            previous = current[-1]
            delta = (point.event_time - previous.event_time).total_seconds()
            if (point.trace_id != previous.trace_id or delta <= 0.0 or
                    delta > config.max_segment_gap_seconds):
                by_device[device_id].append(current)
                current = [point]
                continue
            current.append(point)
        if current:
            by_device[device_id].append(current)
    return by_device


def device_dt_values(points: list[AlignedPoint]) -> list[float]:
    """统计同一设备相邻点之间的时间间隔。

    Args:
        points: 点序列。

    Returns:
        list[float]: 设备内部相邻点的时间间隔列表。
    """
    values: list[float] = []
    for previous, current in zip(points, points[1:]):
        delta = (current.event_time - previous.event_time).total_seconds()
        if delta > 0.0:
            values.append(delta)
    return values
