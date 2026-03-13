"""burst 路径相关逻辑的门面模块。"""

from __future__ import annotations

from drone_kalman_filter._prefilter_burst_core import (
    find_future_anchor,
    repair_points_burst,
)
from drone_kalman_filter._prefilter_fusion import (
    FUSION_DEVICE_ID,
    FUSION_JUMP_DISTANCE_M,
    FUSION_JUMP_SPEED_MPS,
    FUSION_MIN_BURST_STEPS,
    FUSION_REQUIRED_STABLE_STEPS,
    FUSION_STABLE_DISTANCE_M,
    FUSION_STABLE_SPEED_MPS,
    repair_points_fusion_micro_burst,
)
from drone_kalman_filter._prefilter_suspicion import (
    classify_suspicion,
    has_burst_candidate,
    is_bridge_spike_from_anchor,
    qualifies_as_burst,
    repair_single_point,
)

__all__ = [
    "FUSION_DEVICE_ID",
    "FUSION_JUMP_DISTANCE_M",
    "FUSION_JUMP_SPEED_MPS",
    "FUSION_MIN_BURST_STEPS",
    "FUSION_REQUIRED_STABLE_STEPS",
    "FUSION_STABLE_DISTANCE_M",
    "FUSION_STABLE_SPEED_MPS",
    "classify_suspicion",
    "find_future_anchor",
    "has_burst_candidate",
    "is_bridge_spike_from_anchor",
    "qualifies_as_burst",
    "repair_points_burst",
    "repair_points_fusion_micro_burst",
    "repair_single_point",
]
