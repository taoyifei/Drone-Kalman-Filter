"""协议消息解析与经纬度回写工具。"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

_PLAIN_NUMBER_PATTERN = re.compile(r"^-?\d+(?:\.(\d+))?$")


@dataclass(frozen=True, slots=True)
class TrackKey:
    """标识一条独立平滑轨迹的主键。"""

    # 主链路按“同一目标 + 同一设备”隔离状态。
    target_id: str
    device_id: str


@dataclass(slots=True)
class ParsedMessage:
    """保存主链路统一使用的解析后消息。"""

    # 这里保存的是主链路真正需要的标准化字段，避免后续逻辑反复翻原始 JSON。
    arrival_seq: int
    track_key: TrackKey | None
    trace_id: str | None
    event_time: datetime | None
    process_time: datetime | None
    latitude: float | None
    longitude: float | None
    latitude_text: str | None
    longitude_text: str | None
    message: dict[str, Any]

    @property
    def has_valid_coordinates(self) -> bool:
        """判断当前消息是否带有可用经纬度。

        Args:
            None. 不接收额外参数。

        Returns:
            bool: 坐标字段是否完整且可用。
        """
        return self.latitude is not None and self.longitude is not None

    @property
    def is_smoothable(self) -> bool:
        """判断当前消息是否满足进入平滑链路的条件。

        Args:
            None. 不接收额外参数。

        Returns:
            bool: 当前消息是否满足平滑条件。
        """
        return (self.track_key is not None and self.event_time is not None and
                self.has_valid_coordinates)

    @property
    def arrival_time(self) -> datetime | None:
        """返回当前消息用于空闲判断的到达时间。

        Args:
            None. 不接收额外参数。

        Returns:
            datetime | None: 可用于释放判断的到达时间；缺失时返回 None。
        """
        return self.process_time or self.event_time


def parse_message(arrival_seq: int, message: dict[str, Any]) -> ParsedMessage:
    """把原始消息解析成主链路统一使用的结构。

    Args:
        arrival_seq: 输入消息的到达序号。
        message: 输入或输出的消息字典。

    Returns:
        ParsedMessage: 标准化后的消息对象。
    """
    # 深拷贝后再改值，保证外部传入的原消息对象不被主链路偷偷修改。
    cloned = copy.deepcopy(message)
    identity = cloned.get("identity") or {}
    source = cloned.get("source") or {}
    spatial = cloned.get("spatial") or {}
    position = spatial.get("position") or {}

    target_id = _read_non_empty(identity.get("targetId"))
    device_id = _read_non_empty(source.get("deviceId"))
    track_key = TrackKey(target_id,
                         device_id) if target_id and device_id else None

    latitude_text = position.get("latitude")
    longitude_text = position.get("longitude")
    latitude = _parse_coordinate(latitude_text, is_latitude=True)
    longitude = _parse_coordinate(longitude_text, is_latitude=False)

    return ParsedMessage(
        arrival_seq=arrival_seq,
        track_key=track_key,
        trace_id=_read_non_empty(identity.get("traceId")),
        event_time=parse_time(cloned.get("eventTime")),
        process_time=parse_time(cloned.get("processTime")),
        latitude=latitude,
        longitude=longitude,
        latitude_text=latitude_text if isinstance(latitude_text, str) else None,
        longitude_text=longitude_text
        if isinstance(longitude_text, str) else None,
        message=cloned,
    )


def parse_time(value: Any) -> datetime | None:
    """把协议里的时间字符串解析为 datetime。

    Args:
        value: 输入值。

    Returns:
        datetime | None: 解析后的时间；无法解析时返回 None。
    """
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def dump_message(message: dict[str, Any]) -> str:
    """把消息字典序列化成单行 JSON。

    Args:
        message: 输入或输出的消息字典。

    Returns:
        str: 序列化后的 JSON 字符串。
    """
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def set_position_strings(parsed: ParsedMessage, latitude: float,
                         longitude: float) -> dict[str, Any]:
    """按原始格式回写平滑后的经纬度字符串。

    Args:
        parsed: 标准化后的消息对象。
        latitude: 纬度值。
        longitude: 经度值。

    Returns:
        dict[str, Any]: 回写坐标字符串后的消息字典。
    """
    position = ((parsed.message.get("spatial") or {}).get("position") or {})
    # 输出时尽量沿用原始字符串的小数位风格，减少下游把结果看成“格式异常”的风险。
    position["latitude"] = format_like(parsed.latitude_text, latitude)
    position["longitude"] = format_like(parsed.longitude_text, longitude)
    return parsed.message


def format_like(reference: str | None,
                value: float,
                default_decimals: int = 6) -> str:
    """尽量按参考字符串的小数位格式化数值。

    Args:
        reference: 用于格式对齐的参考字符串。
        value: 输入值。
        default_decimals: 默认保留的小数位数。

    Returns:
        str: 参照输入格式格式化后的字符串。
    """
    decimals = extract_decimal_places(reference)
    if decimals is None:
        decimals = default_decimals

    zero_threshold = 0.5 * (10**(-decimals))
    safe_value = 0.0 if abs(value) < zero_threshold else value
    return f"{safe_value:.{decimals}f}"


def extract_decimal_places(reference: str | None) -> int | None:
    """提取原始字符串中的小数位数。

    Args:
        reference: 用于格式对齐的参考字符串。

    Returns:
        int | None: 提取到的小数位数；无法判断时返回 None。
    """
    if not reference:
        return None
    match = _PLAIN_NUMBER_PATTERN.match(reference.strip())
    if match is None:
        return None
    return len(match.group(1) or "")


def _parse_coordinate(value: Any, *, is_latitude: bool) -> float | None:
    """解析并校验单个经纬度字段。

    Args:
        value: 输入值。
        is_latitude: 是否按纬度字段解析。

    Returns:
        float | None: 计算结果；不可用时返回 None。
    """
    if value is None or value == "":
        return None
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return None

    lower, upper = (-90.0, 90.0) if is_latitude else (-180.0, 180.0)
    if coordinate < lower or coordinate > upper:
        return None
    return coordinate


def _read_non_empty(value: Any) -> str | None:
    """读取非空字符串字段。

    Args:
        value: 输入值。

    Returns:
        str | None: 计算结果；不可用时返回 None。
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
