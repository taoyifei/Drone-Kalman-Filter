"""指标计算使用的基础统计与几何工具。"""

from __future__ import annotations

import math
from statistics import median
from typing import Any


def as_float(value: Any) -> float | None:
    """把输入尽量转换成浮点数。

    Args:
        value: 输入值。

    Returns:
        float | None: 转换后的浮点值；无法转换时返回 None。
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_str(value: Any) -> str | None:
    """把输入尽量转换成非空字符串。

    Args:
        value: 输入值。

    Returns:
        str | None: 清洗后的字符串；不可用时返回 None。
    """
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
    """计算一组数值的常用分位统计。

    Args:
        values: 统计值序列。
        include_max: 是否在统计结果中包含最大值。
        treat_as_int: 是否按整数结果输出。

    Returns:
        dict[str, float | int | None]: 生成的结果映射。
    """
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
    """生成旧版 report 使用的简化统计结果。

    Args:
        values: 统计值序列。

    Returns:
        dict[str, float | int | None]: 兼容旧版格式的统计结果。
    """
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
    """按指标类型对统计值做统一取整。

    Args:
        value: 输入值。
        treat_as_int: 是否按整数结果输出。

    Returns:
        float | int: 按配置四舍五入后的统计值。
    """
    if treat_as_int:
        return int(math.ceil(value))
    return round(value, 4)


def percentile(values: list[float], probability: float) -> float:
    """用线性插值计算分位数。

    Args:
        values: 统计值序列。
        probability: 分位点概率，取值范围为 0 到 1。

    Returns:
        float: 对应分位点的值。
    """
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点经纬度之间的大圆距离。

    Args:
        lat1: 第一个点的纬度。
        lon1: 第一个点的经度。
        lat2: 第二个点的纬度。
        lon2: 第二个点的经度。

    Returns:
        float: 两个经纬度点之间的球面距离，单位为米。
    """
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
    """把两点经纬度差近似换算成平面位移向量。

    Args:
        lat1: 第一个点的纬度。
        lon1: 第一个点的经度。
        lat2: 第二个点的纬度。
        lon2: 第二个点的经度。

    Returns:
        tuple[float, float]: 从起点指向终点的局部向量。
    """
    east = haversine_m(lat1, lon1, lat1, lon2)
    north = haversine_m(lat1, lon1, lat2, lon1)
    if lon2 < lon1:
        east *= -1.0
    if lat2 < lat1:
        north *= -1.0
    return east, north


def vector_angle_deg(left: tuple[float, float], right: tuple[float,
                                                             float]) -> float:
    """计算两个二维向量的夹角。

    Args:
        left: 左侧输入值。
        right: 右侧输入值。

    Returns:
        float: 两个向量之间的夹角，单位为度。
    """
    numerator = left[0] * right[0] + left[1] * right[1]
    denominator = math.hypot(*left) * math.hypot(*right)
    if denominator == 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, numerator / denominator))
    return math.degrees(math.acos(cosine))
