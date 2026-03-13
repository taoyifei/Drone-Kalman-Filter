"""鲁棒预处理用到的局部几何工具。"""

from __future__ import annotations

from statistics import median

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.message import ParsedMessage
from drone_kalman_filter._prefilter.types import LocalPoint


def interpolate_point(
    *,
    left_point: LocalPoint,
    right_point: LocalPoint,
    left_time,
    current_time,
    right_time,
) -> LocalPoint:
    """按时间比例在两个锚点之间插值。

    Args:
        left_point: 左侧参考点。
        right_point: 右侧参考点。
        left_time: 左侧参考时间。
        current_time: 当前参考时间。
        right_time: 右侧参考时间。

    Returns:
        LocalPoint: 按时间插值得到的局部点。
    """
    total_seconds = (right_time - left_time).total_seconds()
    if total_seconds <= 0:
        fraction = 0.5
    else:
        fraction = (current_time - left_time).total_seconds() / total_seconds
        fraction = max(0.0, min(1.0, fraction))

    return LocalPoint(
        east_m=left_point.east_m +
        (right_point.east_m - left_point.east_m) * fraction,
        north_m=left_point.north_m +
        (right_point.north_m - left_point.north_m) * fraction,
    )


def extrapolate_point(
    *,
    left_point: LocalPoint,
    right_point: LocalPoint,
    left_time,
    right_time,
    current_time,
    config: PluginConfig,
) -> LocalPoint:
    """按最近一段速度对下一点做外推。

    Args:
        left_point: 左侧参考点。
        right_point: 右侧参考点。
        left_time: 左侧参考时间。
        right_time: 右侧参考时间。
        current_time: 当前参考时间。
        config: 插件配置。

    Returns:
        LocalPoint: 按时间外推得到的局部点。
    """
    previous_dt = max((right_time - left_time).total_seconds(),
                      config.min_dt_seconds)
    current_dt = max((current_time - right_time).total_seconds(),
                     config.min_dt_seconds)
    east_velocity = (right_point.east_m - left_point.east_m) / previous_dt
    north_velocity = (right_point.north_m - left_point.north_m) / previous_dt
    return LocalPoint(
        east_m=right_point.east_m + east_velocity * current_dt,
        north_m=right_point.north_m + north_velocity * current_dt,
    )


def median_filter_points(points: list[LocalPoint],
                         window_size: int) -> list[LocalPoint]:
    """对局部平面点做中值滤波。

    Args:
        points: 点序列。
        window_size: 窗口大小。

    Returns:
        list[LocalPoint]: 中值滤波后的点序列。
    """
    if window_size <= 1 or len(points) <= 2:
        return points

    radius = window_size // 2
    filtered: list[LocalPoint] = []
    for index in range(len(points)):
        start = max(0, index - radius)
        stop = min(len(points), index + radius + 1)
        east_values = [point.east_m for point in points[start:stop]]
        north_values = [point.north_m for point in points[start:stop]]
        filtered.append(
            LocalPoint(east_m=float(median(east_values)),
                       north_m=float(median(north_values))))
    return filtered


def distance(left: LocalPoint, right: LocalPoint) -> float:
    """计算两个局部平面点之间的距离。

    Args:
        left: 左侧输入值。
        right: 右侧输入值。

    Returns:
        float: 两个局部点之间的欧氏距离，单位为米。
    """
    east_delta = right.east_m - left.east_m
    north_delta = right.north_m - left.north_m
    return (east_delta * east_delta + north_delta * north_delta)**0.5


def vector(left: LocalPoint, right: LocalPoint) -> tuple[float, float]:
    """计算从左点指向右点的二维向量。

    Args:
        left: 左侧输入值。
        right: 右侧输入值。

    Returns:
        tuple[float, float]: 从左点指向右点的向量。
    """
    return right.east_m - left.east_m, right.north_m - left.north_m


def vector_norm(value: tuple[float, float]) -> float:
    """计算二维向量的模长。

    Args:
        value: 输入值。

    Returns:
        float: 向量模长。
    """
    east_delta, north_delta = value
    return (east_delta * east_delta + north_delta * north_delta)**0.5


def dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    """计算两个二维向量的点积。

    Args:
        left: 左侧输入值。
        right: 右侧输入值。

    Returns:
        float: 点积结果。
    """
    return left[0] * right[0] + left[1] * right[1]


def dt_seconds(previous: ParsedMessage, current: ParsedMessage,
               config: PluginConfig) -> float:
    """计算两个观测之间的有效时间间隔。

    Args:
        previous: 前一个观测或时间点。
        current: 当前观测或时间点。
        config: 插件配置。

    Returns:
        float: 满足最小步长约束后的时间间隔。
    """
    return max((current.event_time - previous.event_time).total_seconds(),
               config.min_dt_seconds)
