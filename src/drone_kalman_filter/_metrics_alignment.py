"""raw 与 smoothed 的逐行对齐和按设备切段。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drone_kalman_filter._metrics_statistics import as_float, as_str
from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import parse_time


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
        return (
            self.raw_latitude is not None
            and self.raw_longitude is not None
            and self.smoothed_latitude is not None
            and self.smoothed_longitude is not None
        )


class AcceptanceError(ValueError):
    """表示 raw 与 smoothed 验收输入无法对齐。"""


def load_aligned_points(
    raw_path: str | Path,
    smoothed_path: str | Path,
) -> list[AlignedPoint]:
    """逐行读取并对齐 raw 与 smoothed 数据。

    Args:
        raw_path: 原始 JSONL 文件路径。
        smoothed_path: 平滑后 JSONL 文件路径。

    Returns:
        list[AlignedPoint]: 按行对齐后的点列表。

    Raises:
        AcceptanceError: 当 raw 与 smoothed 文件无法逐行对齐时抛出。
    """
    raw_lines, smoothed_lines = _read_alignment_lines(raw_path, smoothed_path)
    points: list[AlignedPoint] = []
    for line_no, raw_payload, smoothed_payload in _iter_aligned_payloads(
        raw_lines,
        smoothed_lines,
    ):
        points.append(
            _build_aligned_point(
                line_no=line_no,
                raw_payload=raw_payload,
                smoothed_payload=smoothed_payload,
            )
        )
    return points


def _read_alignment_lines(
    raw_path: str | Path,
    smoothed_path: str | Path,
) -> tuple[list[str], list[str]]:
    """读取 raw 与 smoothed 的全部文本行。

    Args:
        raw_path: 原始 JSONL 文件路径。
        smoothed_path: 平滑后 JSONL 文件路径。

    Returns:
        tuple[list[str], list[str]]: raw 与 smoothed 的文本行列表。

    Raises:
        AcceptanceError: 当两份文件行数不一致时抛出。
    """
    raw_lines = Path(raw_path).read_text(encoding="utf-8").splitlines()
    smoothed_lines = Path(smoothed_path).read_text(
        encoding="utf-8",
    ).splitlines()
    if len(raw_lines) != len(smoothed_lines):
        raise AcceptanceError(
            "Raw and smoothed files have different line counts.",
        )
    return raw_lines, smoothed_lines


def _iter_aligned_payloads(
    raw_lines: list[str],
    smoothed_lines: list[str],
) -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
    """逐行解析 raw 与 smoothed payload。

    Args:
        raw_lines: 原始文件的文本行列表。
        smoothed_lines: 平滑文件的文本行列表。

    Returns:
        tuple[int, dict, dict]: 行号、raw payload 和 smoothed payload。
    """
    for line_no, (raw_line, smoothed_line) in enumerate(
        zip(raw_lines, smoothed_lines),
        start=1,
    ):
        yield line_no, json.loads(raw_line), json.loads(smoothed_line)


def _build_aligned_point(
    *,
    line_no: int,
    raw_payload: dict,
    smoothed_payload: dict,
) -> AlignedPoint:
    """构造单行的对齐结果。

    Args:
        line_no: 当前行号。
        raw_payload: 原始 JSON payload。
        smoothed_payload: 平滑 JSON payload。

    Returns:
        AlignedPoint: 当前行构造出的对齐点。

    Raises:
        AcceptanceError: 当当前行无法完成对齐时抛出。
    """
    raw_identity = raw_payload.get("identity") or {}
    raw_source = raw_payload.get("source") or {}
    smoothed_identity = smoothed_payload.get("identity") or {}
    smoothed_source = smoothed_payload.get("source") or {}
    _validate_alignment_keys(
        line_no=line_no,
        raw_payload=raw_payload,
        smoothed_payload=smoothed_payload,
        raw_identity=raw_identity,
        raw_source=raw_source,
        smoothed_identity=smoothed_identity,
        smoothed_source=smoothed_source,
    )
    raw_position = _position_payload(raw_payload)
    smoothed_position = _position_payload(smoothed_payload)
    event_time = _parse_event_time(line_no, raw_payload.get("eventTime"))
    smoothed_latitude, smoothed_longitude = _parse_smoothed_coordinates(
        line_no,
        smoothed_position,
    )
    return AlignedPoint(
        line_no=line_no,
        target_id=as_str(raw_identity.get("targetId")),
        device_id=as_str(raw_source.get("deviceId")),
        trace_id=as_str(raw_identity.get("traceId")),
        event_time_text=raw_payload.get("eventTime"),
        event_time=event_time,
        raw_latitude=as_float(raw_position.get("latitude")),
        raw_longitude=as_float(raw_position.get("longitude")),
        smoothed_latitude=smoothed_latitude,
        smoothed_longitude=smoothed_longitude,
    )


def _validate_alignment_keys(
    *,
    line_no: int,
    raw_payload: dict,
    smoothed_payload: dict,
    raw_identity: dict,
    raw_source: dict,
    smoothed_identity: dict,
    smoothed_source: dict,
) -> None:
    """校验当前行的 raw 与 smoothed 对齐键是否一致。

    Args:
        line_no: 当前行号。
        raw_payload: 原始 JSON payload。
        smoothed_payload: 平滑 JSON payload。
        raw_identity: 原始 identity 字段。
        raw_source: 原始 source 字段。
        smoothed_identity: 平滑 identity 字段。
        smoothed_source: 平滑 source 字段。

    Returns:
        None: 不返回值。

    Raises:
        AcceptanceError: 当对齐键不一致时抛出。
    """
    raw_key = _alignment_key(raw_identity, raw_source, raw_payload)
    smoothed_key = _alignment_key(
        smoothed_identity,
        smoothed_source,
        smoothed_payload,
    )
    if raw_key != smoothed_key:
        raise AcceptanceError(f"Alignment mismatch at line {line_no}.")


def _alignment_key(
    identity: dict,
    source: dict,
    payload: dict,
) -> tuple[object, object, object, object]:
    """构造用于逐行对齐的键。

    Args:
        identity: identity 字段。
        source: source 字段。
        payload: 当前 JSON payload。

    Returns:
        tuple[object, object, object, object]: 对齐比较用的键。
    """
    return (
        identity.get("targetId"),
        source.get("deviceId"),
        identity.get("traceId"),
        payload.get("eventTime"),
    )


def _position_payload(payload: dict) -> dict:
    """提取 payload 中的 position 字段。

    Args:
        payload: 当前 JSON payload。

    Returns:
        dict: position 字段或空字典。
    """
    return ((payload.get("spatial") or {}).get("position") or {})


def _parse_event_time(line_no: int, event_time_text: object) -> object:
    """解析当前行的 eventTime。

    Args:
        line_no: 当前行号。
        event_time_text: eventTime 原始字段。

    Returns:
        object: 解析后的时间对象。

    Raises:
        AcceptanceError: 当 eventTime 无法解析时抛出。
    """
    event_time = parse_time(event_time_text)
    if event_time is None:
        raise AcceptanceError(f"Invalid eventTime at line {line_no}.")
    return event_time


def _parse_smoothed_coordinates(
    line_no: int,
    smoothed_position: dict,
) -> tuple[float, float]:
    """解析当前行的 smoothed 经度和纬度。

    Args:
        line_no: 当前行号。
        smoothed_position: smoothed 的 position 字段。

    Returns:
        tuple[float, float]: 解析后的纬度和经度。

    Raises:
        AcceptanceError: 当 smoothed 坐标无效时抛出。
    """
    smoothed_latitude = as_float(smoothed_position.get("latitude"))
    smoothed_longitude = as_float(smoothed_position.get("longitude"))
    if smoothed_latitude is None or smoothed_longitude is None:
        raise AcceptanceError(
            f"Invalid smoothed coordinates at line {line_no}.",
        )
    return smoothed_latitude, smoothed_longitude


def group_points_by_device(
    points: list[AlignedPoint],
) -> dict[str, list[AlignedPoint]]:
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
    points: list[AlignedPoint],
) -> dict[tuple[str, str], list[AlignedPoint]]:
    """按 targetId 和 deviceId 分组有效点。

    Args:
        points: 点序列。

    Returns:
        dict[tuple[str, str], list[AlignedPoint]]: 按轨迹分组后的点集合。
    """
    grouped: dict[tuple[str, str], list[AlignedPoint]] = defaultdict(list)
    for point in points:
        if (
            point.device_id is None
            or point.target_id is None
            or not point.has_valid_coordinates
        ):
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
