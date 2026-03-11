"""Kalman 与 RTS 平滑的公开入口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.geo import LocalTangentPlane
from drone_kalman_filter.message import ParsedMessage
from drone_kalman_filter._kalman_rts import backward_smooth, forward_filter


@dataclass(frozen=True, slots=True)
class SmoothedPosition:
    """表示平滑后的局部二维位置。"""

    east_m: float
    north_m: float


def smooth_positions(
    observations: Sequence[ParsedMessage],
    plane: LocalTangentPlane,
    config: PluginConfig,
) -> list[SmoothedPosition]:
    """对一段观测做前向 Kalman 和后向 RTS 平滑。"""
    if not observations:
        return []

    if len(observations) == 1:
        east, north = plane.to_local(observations[0].latitude,
                                     observations[0].longitude)
        return [SmoothedPosition(east, north)]

    measurements = np.array(
        [
            plane.to_local(item.latitude, item.longitude)
            for item in observations
        ],
        dtype=float,
    )
    filter_pass = forward_filter(observations, measurements, config)
    smoothed_states = backward_smooth(filter_pass)

    return [
        SmoothedPosition(float(state[0]), float(state[1]))
        for state in smoothed_states
    ]
