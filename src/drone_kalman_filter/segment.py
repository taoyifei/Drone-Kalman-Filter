from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import PluginConfig
from .geo import LocalTangentPlane
from .kalman import smooth_positions
from .message import ParsedMessage, set_position_strings


@dataclass(slots=True)
class BufferedObservation:
    parsed: ParsedMessage
    emitted: bool = False


class SegmentSmoother:
    def __init__(self, trace_id: str | None, config: PluginConfig, anchor_latitude: float, anchor_longitude: float) -> None:
        self.trace_id = trace_id
        self.config = config
        self.plane = LocalTangentPlane(anchor_latitude, anchor_longitude)
        # 这里维护固定滞后窗口；只有窗口中“已经成熟”的点才会对外释放。
        self.buffer: deque[BufferedObservation] = deque()

    def append(self, parsed: ParsedMessage) -> list[tuple[int, dict]]:
        while len(self.buffer) >= self.config.window_size and self.buffer and self.buffer[0].emitted:
            self.buffer.popleft()

        self.buffer.append(BufferedObservation(parsed=parsed))
        return self._emit_mature_observation()

    def flush(self) -> list[tuple[int, dict]]:
        if not self.buffer:
            return []

        # 流结束或切段时，把缓冲里剩余点全部补齐输出，保证输入输出条数最终一致。
        smoothed_positions = smooth_positions([item.parsed for item in self.buffer], self.plane, self.config)
        outputs: list[tuple[int, dict]] = []
        for index, item in enumerate(self.buffer):
            if item.emitted:
                continue
            item.emitted = True
            outputs.append(self._materialize(index, smoothed_positions[index]))

        self.buffer.clear()
        return outputs

    def _emit_mature_observation(self) -> list[tuple[int, dict]]:
        if len(self.buffer) <= self.config.lag_points:
            return []

        smoothed_positions = smooth_positions([item.parsed for item in self.buffer], self.plane, self.config)
        # 只发布距离窗口尾部 lag_points 之外的那个点，形成固定滞后输出。
        candidate_index = len(self.buffer) - self.config.lag_points - 1
        candidate = self.buffer[candidate_index]
        if candidate.emitted:
            return []

        candidate.emitted = True
        return [self._materialize(candidate_index, smoothed_positions[candidate_index])]

    def _materialize(self, index: int, smoothed_position) -> tuple[int, dict]:
        item = self.buffer[index]
        latitude, longitude = self.plane.to_geodetic(smoothed_position.east_m, smoothed_position.north_m)
        message = set_position_strings(item.parsed, latitude, longitude)
        return item.parsed.arrival_seq, message
