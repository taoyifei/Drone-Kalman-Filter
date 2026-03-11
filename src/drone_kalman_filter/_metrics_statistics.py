"""指标计算使用的基础统计与几何工具。"""

from __future__ import annotations

import math
from statistics import median
from typing import Any


def as_float(value: Any) -> float | None:
    """把输入尽量转换成浮点数。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_str(value: Any) -> str | None:
    """把输入尽量转换成非空字符串。"""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def distribution(
    values: list[float] | list[int],
    *,
    include_max: bool,
    treat_as_int: bool = False,
) -> dict[str, float | int | None]:
    """计算一组数值的常用分位统计。"""
    if not values:
        base = {
            "count": 0,
            "median": None,
            "p75": None,
            "p95": None,
            "p99": None
        }
        if include_max:
            base["max"] = None
        return base

    ordered = sorted(float(value) for value in values)
    result: dict[str, float | int | None] = {
        "count":
            len(ordered),
        "median":
            round_stat(percentile(ordered, 0.50), treat_as_int=treat_as_int),
        "p75":
            round_stat(percentile(ordered, 0.75), treat_as_int=treat_as_int),
        "p95":
            round_stat(percentile(ordered, 0.95), treat_as_int=treat_as_int),
        "p99":
            round_stat(percentile(ordered, 0.99), treat_as_int=treat_as_int),
    }
    if include_max:
        result["max"] = round_stat(max(ordered), treat_as_int=treat_as_int)
    return result


def legacy_stats(values: list[float]) -> dict[str, float | int | None]:
    """生成旧版 report 使用的简化统计结果。"""
    if not values:
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None
        }

    return {
        "count": len(values),
        "median": round(median(values), 4),
        "p95": round(percentile(values, 0.95), 4),
        "p99": round(percentile(values, 0.99), 4),
        "max": round(max(values), 4),
    }


def round_stat(value: float, *, treat_as_int: bool) -> float | int:
    """按指标类型对统计值做统一取整。"""
    if treat_as_int:
        return int(math.ceil(value))
    return round(value, 4)


def percentile(values: list[float], probability: float) -> float:
    """用线性插值计算分位数。"""
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点经纬度之间的大圆距离。"""
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(
        delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(
            delta_lambda / 2.0)**2
    return 2.0 * radius * math.asin(math.sqrt(a))


def vector_m(lat1: float, lon1: float, lat2: float,
             lon2: float) -> tuple[float, float]:
    """把两点经纬度差近似换算成平面位移向量。"""
    east = haversine_m(lat1, lon1, lat1, lon2)
    north = haversine_m(lat1, lon1, lat2, lon1)
    if lon2 < lon1:
        east *= -1.0
    if lat2 < lat1:
        north *= -1.0
    return east, north


def vector_angle_deg(left: tuple[float, float], right: tuple[float,
                                                             float]) -> float:
    """计算两个二维向量的夹角。"""
    numerator = left[0] * right[0] + left[1] * right[1]
    denominator = math.hypot(*left) * math.hypot(*right)
    if denominator == 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, numerator / denominator))
    return math.degrees(math.acos(cosine))
