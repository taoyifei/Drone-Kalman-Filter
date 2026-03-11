"""鲁棒预处理内部使用的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class LocalPoint:
    """表示局部平面中的二维点。"""

    east_m: float
    north_m: float


@dataclass(frozen=True, slots=True)
class Suspicion:
    """保存单个点的异常分类结果。"""

    hard_jump: bool
    bridge_spike: bool

    @property
    def is_suspicious(self) -> bool:
        """判断当前分类结果是否需要触发修复。"""
        return self.hard_jump or self.bridge_spike


@dataclass(frozen=True, slots=True)
class BurstAnchor:
    """表示 burst 修复时找到的未来锚点。"""

    repair_end: int
    anchor_index: int | None


@dataclass(frozen=True, slots=True)
class TrustedAnchor:
    """表示跨窗口延续的可信锚点。"""

    point: LocalPoint
    event_time: datetime
