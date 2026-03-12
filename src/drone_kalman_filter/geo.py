"""经纬度与局部平面坐标互转。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True, slots=True)
class LocalTangentPlane:
    """表示以首点为原点的局部切平面。"""

    origin_lat_deg: float
    origin_lon_deg: float
    _origin_lat_rad: float = field(init=False, repr=False)
    _cos_origin_lat: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """预计算局部坐标转换需要的常量。

        Args:
            None. 不接收额外参数。

        Returns:
            None: 不返回值。
        """
        object.__setattr__(self, "_origin_lat_rad",
                           math.radians(self.origin_lat_deg))
        object.__setattr__(self, "_cos_origin_lat",
                           math.cos(self._origin_lat_rad))

    def to_local(self, latitude_deg: float,
                 longitude_deg: float) -> tuple[float, float]:
        """把经纬度转换成局部平面坐标。

        Args:
            latitude_deg: 纬度，单位为度。
            longitude_deg: 经度，单位为度。

        Returns:
            tuple[float, float]: 对应的局部东、北向坐标。
        """
        # 不直接在经纬度上做滤波，先映射到局部平面，数值会更稳定。
        east = math.radians(longitude_deg - self.origin_lon_deg
                           ) * EARTH_RADIUS_M * self._cos_origin_lat
        north = math.radians(latitude_deg -
                             self.origin_lat_deg) * EARTH_RADIUS_M
        return east, north

    def to_geodetic(self, east_m: float, north_m: float) -> tuple[float, float]:
        """把局部平面坐标还原成经纬度。

        Args:
            east_m: 局部平面东向坐标，单位为米。
            north_m: 局部平面北向坐标，单位为米。

        Returns:
            tuple[float, float]: 对应的地理坐标。
        """
        latitude_deg = self.origin_lat_deg + math.degrees(
            north_m / EARTH_RADIUS_M)
        longitude_scale = EARTH_RADIUS_M * self._cos_origin_lat
        if longitude_scale == 0.0:
            longitude_deg = self.origin_lon_deg
        else:
            longitude_deg = self.origin_lon_deg + math.degrees(
                east_m / longitude_scale)
        return latitude_deg, longitude_deg
