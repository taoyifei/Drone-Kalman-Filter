from __future__ import annotations

import math
from dataclasses import dataclass, field


EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True, slots=True)
class LocalTangentPlane:
    origin_lat_deg: float
    origin_lon_deg: float
    _origin_lat_rad: float = field(init=False, repr=False)
    _cos_origin_lat: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_origin_lat_rad", math.radians(self.origin_lat_deg))
        object.__setattr__(self, "_cos_origin_lat", math.cos(self._origin_lat_rad))

    def to_local(self, latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
        # 不直接在经纬度上做滤波，先映射到局部平面，数值会更稳定。
        east = math.radians(longitude_deg - self.origin_lon_deg) * EARTH_RADIUS_M * self._cos_origin_lat
        north = math.radians(latitude_deg - self.origin_lat_deg) * EARTH_RADIUS_M
        return east, north

    def to_geodetic(self, east_m: float, north_m: float) -> tuple[float, float]:
        latitude_deg = self.origin_lat_deg + math.degrees(north_m / EARTH_RADIUS_M)
        longitude_scale = EARTH_RADIUS_M * self._cos_origin_lat
        if longitude_scale == 0.0:
            longitude_deg = self.origin_lon_deg
        else:
            longitude_deg = self.origin_lon_deg + math.degrees(east_m / longitude_scale)
        return latitude_deg, longitude_deg
