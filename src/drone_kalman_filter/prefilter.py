"""带鲁棒预处理的单段平滑门面。"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.geo import LocalTangentPlane
from drone_kalman_filter.kalman import SmoothedPosition, smooth_positions
from drone_kalman_filter.message import ParsedMessage, set_position_strings
from drone_kalman_filter.segment import BufferedObservation
from drone_kalman_filter._prefilter_burst import (
    has_burst_candidate,
    repair_points_burst,
)
from drone_kalman_filter._prefilter_geometry import (
    distance,
    median_filter_points,
)
from drone_kalman_filter._prefilter_legacy import repair_points_legacy
from drone_kalman_filter._prefilter_types import LocalPoint, TrustedAnchor


class RobustPrefilterSegmentSmoother:
    """组合预处理与 fixed-lag Kalman 的单段平滑器。"""

    def __init__(self, trace_id: str | None, config: PluginConfig,
                 anchor_latitude: float, anchor_longitude: float) -> None:
        """初始化带鲁棒预处理的单段平滑器。

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
        self.buffer: deque[BufferedObservation] = deque()
        # trusted_anchor 是跨窗口延续的“可信锚点”，burst 修复时只能围绕它做修补。
        self._trusted_anchor: TrustedAnchor | None = None
        self._burst_cooldown = 0

    def append(self, parsed: ParsedMessage) -> list[tuple[int, dict]]:
        """追加一个点并尝试输出成熟结果。

        Args:
            parsed: 标准化后的消息对象。

        Returns:
            list[tuple[int, dict]]: 当前已成熟的输出序号与消息列表。
        """
        while (len(self.buffer) >= self.config.prefilter_window_size and
               self.buffer and self.buffer[0].emitted):
            self.buffer.popleft()

        self.buffer.append(BufferedObservation(parsed=parsed))
        return self._emit_mature_observation()

    def flush(self) -> list[tuple[int, dict]]:
        """在切段或流结束时补齐所有剩余输出。

        Args:
            None. 不接收额外参数。

        Returns:
            list[tuple[int, dict]]: 当前缓冲区剩余的输出序号与消息列表。
        """
        if not self.buffer:
            return []

        (
            repaired_observations,
            repaired_points,
            anchorable_flags,
            used_burst_path,
            smoothed_positions,
        ) = self._compute_repaired_and_smoothed()
        outputs: list[tuple[int, dict]] = []
        for index, item in enumerate(self.buffer):
            if item.emitted:
                continue
            item.emitted = True
            if anchorable_flags[index]:
                self._trusted_anchor = TrustedAnchor(
                    point=repaired_points[index],
                    event_time=repaired_observations[index].event_time,
                )
            outputs.append(self._materialize(index, smoothed_positions[index]))
        self._update_burst_cooldown(used_burst_path)

        self.buffer.clear()
        return outputs

    def _emit_mature_observation(self) -> list[tuple[int, dict]]:
        """按固定滞后规则释放一个成熟点。

        Args:
            None. 不接收额外参数。

        Returns:
            list[tuple[int, dict]]: 当前已成熟的输出序号与消息列表。
        """
        if len(self.buffer) <= self.config.prefilter_lag_points:
            return []

        (
            repaired_observations,
            repaired_points,
            anchorable_flags,
            used_burst_path,
            smoothed_positions,
        ) = self._compute_repaired_and_smoothed()
        candidate_index = len(
            self.buffer) - self.config.prefilter_lag_points - 1
        candidate = self.buffer[candidate_index]
        if candidate.emitted:
            return []

        candidate.emitted = True
        if anchorable_flags[candidate_index]:
            self._trusted_anchor = TrustedAnchor(
                point=repaired_points[candidate_index],
                event_time=repaired_observations[candidate_index].event_time,
            )
        self._update_burst_cooldown(used_burst_path)
        return [
            self._materialize(candidate_index,
                              smoothed_positions[candidate_index])
        ]

    def _compute_repaired_and_smoothed(
        self,
    ) -> tuple[list[ParsedMessage], list[LocalPoint], list[bool], bool,
               list[SmoothedPosition]]:
        """生成修复后的观测并完成窗口内平滑。

        Args:
            None. 不接收额外参数。

        Returns:
            tuple[list[ParsedMessage], list[LocalPoint], list[bool], bool, list[SmoothedPosition]]: 修复后的观测、修复点、可锚定标记、是否走
                                                                                                    burst
                                                                                                    路径以及平滑结果。
        """
        (
            repaired_observations,
            repaired_points,
            anchorable_flags,
            used_burst_path,
        ) = self._repaired_observations()
        smoothed_positions = smooth_positions(repaired_observations, self.plane,
                                              self.config)
        return (
            repaired_observations,
            repaired_points,
            anchorable_flags,
            used_burst_path,
            smoothed_positions,
        )

    def _repaired_observations(
            self
    ) -> tuple[list[ParsedMessage], list[LocalPoint], list[bool], bool]:
        """根据当前窗口生成修复后的观测序列。

        Args:
            None. 不接收额外参数。

        Returns:
            tuple[list[ParsedMessage], list[LocalPoint], list[bool], bool]: 修复后的观测、输出点、可锚定标记以及是否走
                                                                            burst
                                                                            路径。
        """
        observations = [item.parsed for item in self.buffer]
        if len(observations) <= 1:
            raw_points = [
                LocalPoint(*self.plane.to_local(item.latitude, item.longitude))
                for item in observations
            ]
            return observations, raw_points, [False] * len(observations), False

        raw_points = [
            LocalPoint(*self.plane.to_local(item.latitude, item.longitude))
            for item in observations
        ]
        used_burst_path = self._burst_cooldown > 0 or has_burst_candidate(
            observations,
            raw_points,
            self.config,
            self._trusted_anchor,
        )
        if used_burst_path:
            (
                repaired_points,
                had_burst_repair,
                altered_flags,
            ) = repair_points_burst(
                observations,
                raw_points,
                self.config,
                seed_anchor=self._trusted_anchor,
            )
            output_points = repaired_points
            if not had_burst_repair:
                output_points = median_filter_points(
                    repaired_points, self.config.prefilter_median_window_size)
            anchorable_flags = [
                (not altered) or distance(raw_point, output_point)
                <= self.config.prefilter_bridge_neighbor_distance_m
                for raw_point, output_point, altered in zip(
                    raw_points, output_points, altered_flags)
            ]
        else:
            repaired_points = repair_points_legacy(observations, raw_points,
                                                   self.config)
            output_points = median_filter_points(
                repaired_points, self.config.prefilter_median_window_size)
            anchorable_flags = [
                distance(raw_point, output_point)
                <= self.config.prefilter_bridge_neighbor_distance_m
                for raw_point, output_point in zip(raw_points, output_points)
            ]

        repaired_observations = []
        for parsed, point in zip(observations, output_points):
            latitude, longitude = self.plane.to_geodetic(
                point.east_m,
                point.north_m,
            )
            repaired_observations.append(
                replace(parsed, latitude=latitude, longitude=longitude))
        return (
            repaired_observations,
            output_points,
            anchorable_flags,
            used_burst_path,
        )

    def _materialize(self, index: int,
                     smoothed_position: SmoothedPosition) -> tuple[int, dict]:
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

    def _update_burst_cooldown(self, used_burst_path: bool) -> None:
        """更新 burst 路径的短期冷却状态。

        Args:
            used_burst_path: 是否采用 burst 修复路径。

        Returns:
            None: 不返回值。
        """
        if used_burst_path:
            self._burst_cooldown = self.config.prefilter_burst_max_run_length
        elif self._burst_cooldown > 0:
            self._burst_cooldown -= 1
