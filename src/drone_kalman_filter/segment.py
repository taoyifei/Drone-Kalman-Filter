"""基础 fixed-lag 单段平滑器。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.geo import LocalTangentPlane
from drone_kalman_filter.kalman import smooth_positions
from drone_kalman_filter.message import ParsedMessage, set_position_strings


@dataclass(slots=True)
class BufferedObservation:
    """缓存单段平滑窗口中的一个观测点。"""

    parsed: ParsedMessage
    emitted: bool = False


class SegmentSmoother:
    """实现基础 fixed-lag 窗口平滑的单段平滑器。"""

    def __init__(self, trace_id: str | None, config: PluginConfig,
                 anchor_latitude: float, anchor_longitude: float) -> None:
        """初始化单段 fixed-lag 平滑器。

        Args:
            trace_id: 轨迹标识。
            config: 插件配置。
            anchor_latitude: 局部切平面锚点的纬度。
            anchor_longitude: 局部切平面锚点的经度。

        Returns:
            None: 不返回值。
        """
        self.trace_id = trace_id
        self.config = config
        self.plane = LocalTangentPlane(anchor_latitude, anchor_longitude)
        # 这里维护固定滞后窗口；只有窗口中“已经成熟”的点才会对外释放。
        self.buffer: deque[BufferedObservation] = deque()

    def append(self, parsed: ParsedMessage) -> list[tuple[int, dict]]:
        """向当前段追加一个点并尝试释放成熟输出。

        Args:
            parsed: 标准化后的消息对象。

        Returns:
            list[tuple[int, dict]]: 当前已成熟的输出序号与消息列表。
        """
        while len(
                self.buffer
        ) >= self.config.window_size and self.buffer and self.buffer[0].emitted:
            self.buffer.popleft()

        self.buffer.append(BufferedObservation(parsed=parsed))
        return self._emit_mature_observation()

    def flush(self) -> list[tuple[int, dict]]:
        """在切段或流结束时补齐缓冲区剩余输出。

        Args:
            None. 不接收额外参数。

        Returns:
            list[tuple[int, dict]]: 当前缓冲区剩余的输出序号与消息列表。
        """
        if not self.buffer:
            return []

        # 流结束或切段时，把缓冲里剩余点全部补齐输出，保证输入输出条数最终一致。
        smoothed_positions = smooth_positions(
            [item.parsed for item in self.buffer], self.plane, self.config)
        outputs: list[tuple[int, dict]] = []
        for index, item in enumerate(self.buffer):
            if item.emitted:
                continue
            item.emitted = True
            outputs.append(self._materialize(index, smoothed_positions[index]))

        self.buffer.clear()
        return outputs

    def _emit_mature_observation(self) -> list[tuple[int, dict]]:
        """按固定滞后规则释放当前窗口中的成熟点。

        Args:
            None. 不接收额外参数。

        Returns:
            list[tuple[int, dict]]: 当前已成熟的输出序号与消息列表。
        """
        if len(self.buffer) <= self.config.lag_points:
            return []

        smoothed_positions = smooth_positions(
            [item.parsed for item in self.buffer], self.plane, self.config)
        # 只发布距离窗口尾部 lag_points 之外的那个点，形成固定滞后输出。
        candidate_index = len(self.buffer) - self.config.lag_points - 1
        candidate = self.buffer[candidate_index]
        if candidate.emitted:
            return []

        candidate.emitted = True
        return [
            self._materialize(candidate_index,
                              smoothed_positions[candidate_index])
        ]

    def _materialize(self, index: int, smoothed_position) -> tuple[int, dict]:
        """把局部平滑结果回写成业务消息。

        Args:
            index: 目标元素在序列或缓冲区中的索引。
            smoothed_position: 平滑后的位置结果。

        Returns:
            tuple[int, dict]: 生成的结果元组。
        """
        item = self.buffer[index]
        latitude, longitude = self.plane.to_geodetic(smoothed_position.east_m,
                                                     smoothed_position.north_m)
        message = set_position_strings(item.parsed, latitude, longitude)
        return item.parsed.arrival_seq, message
