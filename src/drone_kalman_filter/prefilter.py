from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from statistics import median

from .config import PluginConfig
from .geo import LocalTangentPlane
from .kalman import SmoothedPosition, smooth_positions
from .message import ParsedMessage, set_position_strings
from .segment import BufferedObservation


@dataclass(frozen=True, slots=True)
class LocalPoint:
    east_m: float
    north_m: float


@dataclass(frozen=True, slots=True)
class _Suspicion:
    hard_jump: bool
    bridge_spike: bool

    @property
    def is_suspicious(self) -> bool:
        return self.hard_jump or self.bridge_spike


@dataclass(frozen=True, slots=True)
class _BurstAnchor:
    repair_end: int
    anchor_index: int | None


@dataclass(frozen=True, slots=True)
class _TrustedAnchor:
    point: LocalPoint
    event_time: datetime


class RobustPrefilterSegmentSmoother:
    def __init__(self, trace_id: str | None, config: PluginConfig, anchor_latitude: float, anchor_longitude: float) -> None:
        self.trace_id = trace_id
        self.config = config
        self.plane = LocalTangentPlane(anchor_latitude, anchor_longitude)
        self.buffer: deque[BufferedObservation] = deque()
        # trusted_anchor 是跨窗口延续的“可信锚点”，burst 修复时只能围绕它做修补。
        self._trusted_anchor: _TrustedAnchor | None = None
        self._burst_cooldown = 0

    def append(self, parsed: ParsedMessage) -> list[tuple[int, dict]]:
        while (
            len(self.buffer) >= self.config.prefilter_window_size
            and self.buffer
            and self.buffer[0].emitted
        ):
            self.buffer.popleft()

        self.buffer.append(BufferedObservation(parsed=parsed))
        return self._emit_mature_observation()

    def flush(self) -> list[tuple[int, dict]]:
        if not self.buffer:
            return []

        repaired_observations, repaired_points, anchorable_flags, used_burst_path, smoothed_positions = self._compute_repaired_and_smoothed()
        outputs: list[tuple[int, dict]] = []
        for index, item in enumerate(self.buffer):
            if item.emitted:
                continue
            item.emitted = True
            if anchorable_flags[index]:
                self._trusted_anchor = _TrustedAnchor(
                    point=repaired_points[index],
                    event_time=repaired_observations[index].event_time,
                )
            outputs.append(self._materialize(index, smoothed_positions[index]))
        self._update_burst_cooldown(used_burst_path)

        self.buffer.clear()
        return outputs

    def _emit_mature_observation(self) -> list[tuple[int, dict]]:
        if len(self.buffer) <= self.config.prefilter_lag_points:
            return []

        repaired_observations, repaired_points, anchorable_flags, used_burst_path, smoothed_positions = self._compute_repaired_and_smoothed()
        candidate_index = len(self.buffer) - self.config.prefilter_lag_points - 1
        candidate = self.buffer[candidate_index]
        if candidate.emitted:
            return []

        candidate.emitted = True
        if anchorable_flags[candidate_index]:
            self._trusted_anchor = _TrustedAnchor(
                point=repaired_points[candidate_index],
                event_time=repaired_observations[candidate_index].event_time,
            )
        self._update_burst_cooldown(used_burst_path)
        return [self._materialize(candidate_index, smoothed_positions[candidate_index])]

    def _compute_repaired_and_smoothed(
        self,
    ) -> tuple[list[ParsedMessage], list[LocalPoint], list[bool], bool, list[SmoothedPosition]]:
        repaired_observations, repaired_points, anchorable_flags, used_burst_path = self._repaired_observations()
        smoothed_positions = smooth_positions(repaired_observations, self.plane, self.config)
        return repaired_observations, repaired_points, anchorable_flags, used_burst_path, smoothed_positions

    def _repaired_observations(self) -> tuple[list[ParsedMessage], list[LocalPoint], list[bool], bool]:
        observations = [item.parsed for item in self.buffer]
        if len(observations) <= 1:
            raw_points = [LocalPoint(*self.plane.to_local(item.latitude, item.longitude)) for item in observations]
            return observations, raw_points, [False] * len(observations), False

        raw_points = [LocalPoint(*self.plane.to_local(item.latitude, item.longitude)) for item in observations]
        # 有连续交替尖刺风险时走 burst 路径；否则继续沿用更温和的 legacy 路径。
        used_burst_path = self._burst_cooldown > 0 or _has_burst_candidate(observations, raw_points, self.config, self._trusted_anchor)
        if used_burst_path:
            repaired_points, had_burst_repair, altered_flags = _repair_points_burst(
                observations,
                raw_points,
                self.config,
                seed_anchor=self._trusted_anchor,
            )
            output_points = repaired_points
            if not had_burst_repair:
                # 进入 burst 路径但实际没修到 burst 时，仍允许保留普通去毛刺能力。
                output_points = _median_filter_points(repaired_points, self.config.prefilter_median_window_size)
            anchorable_flags = [
                (not altered) or _distance(raw_point, output_point) <= self.config.prefilter_bridge_neighbor_distance_m
                for raw_point, output_point, altered in zip(raw_points, output_points, altered_flags)
            ]
        else:
            repaired_points = _repair_points_legacy(observations, raw_points, self.config)
            output_points = _median_filter_points(repaired_points, self.config.prefilter_median_window_size)
            anchorable_flags = [
                _distance(raw_point, output_point) <= self.config.prefilter_bridge_neighbor_distance_m
                for raw_point, output_point in zip(raw_points, output_points)
            ]

        repaired_observations: list[ParsedMessage] = []
        for parsed, point in zip(observations, output_points):
            latitude, longitude = self.plane.to_geodetic(point.east_m, point.north_m)
            repaired_observations.append(replace(parsed, latitude=latitude, longitude=longitude))
        return repaired_observations, output_points, anchorable_flags, used_burst_path

    def _materialize(self, index: int, smoothed_position: SmoothedPosition) -> tuple[int, dict]:
        item = self.buffer[index]
        latitude, longitude = self.plane.to_geodetic(smoothed_position.east_m, smoothed_position.north_m)
        message = set_position_strings(item.parsed, latitude, longitude)
        return item.parsed.arrival_seq, message

    def _update_burst_cooldown(self, used_burst_path: bool) -> None:
        if used_burst_path:
            self._burst_cooldown = self.config.prefilter_burst_max_run_length
        elif self._burst_cooldown > 0:
            self._burst_cooldown -= 1


def _has_burst_candidate(
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    seed_anchor: _TrustedAnchor | None,
) -> bool:
    # 这里只做“是否值得切到 burst 修复”的轻量判断，不直接改任何点。
    if len(raw_points) < 3:
        return False

    if seed_anchor is None:
        last_trusted_point = raw_points[0]
        last_trusted_time = observations[0].event_time
        index = 1
    else:
        last_trusted_point = seed_anchor.point
        last_trusted_time = seed_anchor.event_time
        index = 0

    while index < len(raw_points):
        suspicion = _classify_suspicion(
            anchor_point=last_trusted_point,
            anchor_time=last_trusted_time,
            index=index,
            observations=observations,
            raw_points=raw_points,
            config=config,
        )
        if not suspicion.is_suspicious:
            last_trusted_point = raw_points[index]
            last_trusted_time = observations[index].event_time
            index += 1
            continue

        run_end = index
        while (
            run_end + 1 < len(raw_points)
            and run_end - index + 1 < config.prefilter_burst_max_run_length
        ):
            local_dt = _dt_seconds(observations[run_end], observations[run_end + 1], config)
            if local_dt > config.prefilter_hard_distance_dt_seconds:
                break
            next_suspicion = _classify_suspicion(
                anchor_point=last_trusted_point,
                anchor_time=last_trusted_time,
                index=run_end + 1,
                observations=observations,
                raw_points=raw_points,
                config=config,
            )
            if not next_suspicion.is_suspicious:
                break
            run_end += 1

        if run_end > index and _qualifies_as_burst(
            anchor_point=last_trusted_point,
            raw_points=raw_points,
            start_index=index,
            run_end=run_end,
        ):
            return True

        index += 1

    return False


def _repair_points_legacy(
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
) -> list[LocalPoint]:
    # legacy 路径专门处理孤立异常点，尽量不影响正常的快速转弯和连续运动。
    repaired = [raw_points[0]]

    for index in range(1, len(raw_points)):
        raw_point = raw_points[index]
        previous_point = repaired[index - 1]
        dt = _dt_seconds(observations[index - 1], observations[index], config)
        distance_to_previous = _distance(previous_point, raw_point)
        implied_speed = distance_to_previous / dt
        hard_jump = (
            implied_speed > config.prefilter_hard_speed_mps
            or (
                dt <= config.prefilter_hard_distance_dt_seconds
                and distance_to_previous > config.prefilter_hard_distance_m
            )
        )
        bridge_spike = implied_speed > config.prefilter_soft_speed_mps and _is_bridge_spike_legacy(
            observations=observations,
            raw_points=raw_points,
            repaired_points=repaired,
            index=index,
            config=config,
        )

        if hard_jump or bridge_spike:
            repaired.append(_repair_single_point_legacy(index, observations, raw_points, repaired, config))
            continue

        repaired.append(raw_point)

    return repaired


def _repair_points_burst(
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
    *,
    seed_anchor: _TrustedAnchor | None = None,
) -> tuple[list[LocalPoint], bool, list[bool]]:
    # burst 路径处理的是“短时间内连续跳、还会反向折返”的坏样本。
    repaired: list[LocalPoint] = []
    repaired_times: list[datetime] = []
    altered_flags: list[bool] = []
    had_burst_repair = False
    if seed_anchor is None:
        repaired.append(raw_points[0])
        repaired_times.append(observations[0].event_time)
        altered_flags.append(False)
        last_trusted_point = raw_points[0]
        last_trusted_time = observations[0].event_time
        index = 1
    else:
        last_trusted_point = seed_anchor.point
        last_trusted_time = seed_anchor.event_time
        index = 0

    while index < len(raw_points):
        suspicion = _classify_suspicion(
            anchor_point=last_trusted_point,
            anchor_time=last_trusted_time,
            index=index,
            observations=observations,
            raw_points=raw_points,
            config=config,
        )
        if not suspicion.is_suspicious:
            repaired.append(raw_points[index])
            repaired_times.append(observations[index].event_time)
            altered_flags.append(False)
            last_trusted_point = raw_points[index]
            last_trusted_time = observations[index].event_time
            index += 1
            continue

        run_end = index
        while (
            run_end + 1 < len(raw_points)
            and run_end - index + 1 < config.prefilter_burst_max_run_length
        ):
            local_dt = _dt_seconds(observations[run_end], observations[run_end + 1], config)
            if local_dt > config.prefilter_hard_distance_dt_seconds:
                break
            next_suspicion = _classify_suspicion(
                anchor_point=last_trusted_point,
                anchor_time=last_trusted_time,
                index=run_end + 1,
                observations=observations,
                raw_points=raw_points,
                config=config,
            )
            if not next_suspicion.is_suspicious:
                break
            run_end += 1

        is_burst = run_end > index and _qualifies_as_burst(
            anchor_point=last_trusted_point,
            raw_points=raw_points,
            start_index=index,
            run_end=run_end,
        )
        if not is_burst:
            # 只有确认不是 burst，才退回单点修复，避免把所有 hard_jump 都压成持有上一点。
            repaired_point = _repair_single_point(
                index,
                observations,
                raw_points,
                repaired,
                repaired_times,
                last_trusted_point,
                last_trusted_time,
                config,
            )
            repaired.append(repaired_point)
            repaired_times.append(observations[index].event_time)
            altered_flags.append(True)
            last_trusted_point = repaired_point
            last_trusted_time = observations[index].event_time
            index += 1
            continue

        anchor = _find_future_anchor(
            start_index=index,
            current_run_end=run_end,
            anchor_point=last_trusted_point,
            anchor_time=last_trusted_time,
            observations=observations,
            raw_points=raw_points,
            config=config,
        )

        if anchor.anchor_index is None:
            # 看不到可靠未来锚点时，宁可短暂持有上一可信点，也不拿坏点继续外推。
            for current_index in range(index, anchor.repair_end + 1):
                repaired.append(last_trusted_point)
                repaired_times.append(observations[current_index].event_time)
                altered_flags.append(True)
            had_burst_repair = True
            index = anchor.repair_end + 1
            continue

        # 找到未来锚点后，用“上一可信点 -> 未来锚点”对整个 burst 做时间插值。
        for current_index in range(index, anchor.anchor_index):
            repaired.append(
                _interpolate_point(
                    left_point=last_trusted_point,
                    right_point=raw_points[anchor.anchor_index],
                    left_time=last_trusted_time,
                    current_time=observations[current_index].event_time,
                    right_time=observations[anchor.anchor_index].event_time,
                )
            )
            repaired_times.append(observations[current_index].event_time)
            altered_flags.append(True)
        repaired.append(raw_points[anchor.anchor_index])
        had_burst_repair = True
        repaired_times.append(observations[anchor.anchor_index].event_time)
        altered_flags.append(False)
        last_trusted_point = raw_points[anchor.anchor_index]
        last_trusted_time = observations[anchor.anchor_index].event_time
        index = anchor.anchor_index + 1

    return repaired, had_burst_repair, altered_flags


def _find_future_anchor(
    *,
    start_index: int,
    current_run_end: int,
    anchor_point: LocalPoint,
    anchor_time: datetime,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
) -> _BurstAnchor:
    repair_end = current_run_end
    search_limit = min(len(raw_points) - 1, start_index + config.prefilter_burst_max_run_length)

    for candidate_index in range(current_run_end + 1, search_limit + 1):
        # 未来锚点必须先对当前可信锚点“看起来正常”，否则继续扩展待修复范围。
        candidate_suspicion = _classify_suspicion(
            anchor_point=anchor_point,
            anchor_time=anchor_time,
            index=candidate_index,
            observations=observations,
            raw_points=raw_points,
            config=config,
        )
        if candidate_suspicion.is_suspicious:
            repair_end = candidate_index
            continue

        if candidate_index < len(raw_points) - 1:
            # 如果还能看到下一个点，就顺手做一次局部确认，避免把刚恢复的一点误当成稳定锚点。
            next_dt = _dt_seconds(observations[candidate_index], observations[candidate_index + 1], config)
            next_distance = _distance(raw_points[candidate_index], raw_points[candidate_index + 1])
            next_speed = next_distance / next_dt
            if next_speed > config.prefilter_hard_speed_mps or (
                next_dt <= config.prefilter_hard_distance_dt_seconds
                and next_distance > config.prefilter_hard_distance_m
            ):
                repair_end = candidate_index
                continue

        return _BurstAnchor(repair_end=repair_end, anchor_index=candidate_index)

    return _BurstAnchor(repair_end=repair_end, anchor_index=None)


def _repair_single_point_legacy(
    index: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    repaired_points: list[LocalPoint],
    config: PluginConfig,
) -> LocalPoint:
    previous_point = repaired_points[index - 1]

    if index < len(raw_points) - 1:
        next_raw_point = raw_points[index + 1]
        if _distance(previous_point, next_raw_point) <= config.prefilter_hard_distance_m:
            return _interpolate_point(
                left_point=previous_point,
                right_point=next_raw_point,
                left_time=observations[index - 1].event_time,
                current_time=observations[index].event_time,
                right_time=observations[index + 1].event_time,
            )

    if index >= 2:
        return _extrapolate_point(
            left_point=repaired_points[index - 2],
            right_point=repaired_points[index - 1],
            left_time=observations[index - 2].event_time,
            right_time=observations[index - 1].event_time,
            current_time=observations[index].event_time,
            config=config,
        )

    return previous_point


def _is_bridge_spike_legacy(
    *,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    repaired_points: list[LocalPoint],
    index: int,
    config: PluginConfig,
) -> bool:
    if index >= len(raw_points) - 1:
        return False

    previous_point = repaired_points[index - 1]
    current_point = raw_points[index]
    next_point = raw_points[index + 1]
    neighbor_distance = _distance(previous_point, next_point)
    current_to_previous = _distance(current_point, previous_point)
    current_to_next = _distance(current_point, next_point)
    if neighbor_distance > config.prefilter_bridge_neighbor_distance_m:
        return False
    if min(current_to_previous, current_to_next) <= config.prefilter_bridge_center_distance_m:
        return False

    left_dt = _dt_seconds(observations[index - 1], observations[index], config)
    right_dt = _dt_seconds(observations[index], observations[index + 1], config)
    return left_dt <= config.prefilter_hard_distance_dt_seconds and right_dt <= config.prefilter_hard_distance_dt_seconds


def _qualifies_as_burst(
    *,
    anchor_point: LocalPoint,
    raw_points: list[LocalPoint],
    start_index: int,
    run_end: int,
) -> bool:
    if run_end <= start_index:
        return False

    first_vector = _vector(anchor_point, raw_points[start_index])
    first_norm = _vector_norm(first_vector)
    if first_norm == 0.0:
        return False

    for current_index in range(start_index + 1, run_end + 1):
        current_vector = _vector(anchor_point, raw_points[current_index])
        current_norm = _vector_norm(current_vector)
        if current_norm == 0.0:
            continue
        cosine = _dot(first_vector, current_vector) / (first_norm * current_norm)
        if cosine < -0.5:
            return True
    return False


def _repair_single_point(
    index: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    repaired_points: list[LocalPoint],
    repaired_times: list[datetime],
    anchor_point: LocalPoint,
    anchor_time: datetime,
    config: PluginConfig,
) -> LocalPoint:
    if index < len(raw_points) - 1:
        next_raw_point = raw_points[index + 1]
        if _distance(anchor_point, next_raw_point) <= config.prefilter_hard_distance_m:
            return _interpolate_point(
                left_point=anchor_point,
                right_point=next_raw_point,
                left_time=anchor_time,
                current_time=observations[index].event_time,
                right_time=observations[index + 1].event_time,
            )

    if repaired_points:
        left_point = repaired_points[-2] if len(repaired_points) >= 2 else anchor_point
        left_time = repaired_times[-2] if len(repaired_times) >= 2 else anchor_time
        return _extrapolate_point(
            left_point=left_point,
            right_point=repaired_points[-1],
            left_time=left_time,
            right_time=repaired_times[-1],
            current_time=observations[index].event_time,
            config=config,
        )

    return anchor_point


def _classify_suspicion(
    *,
    anchor_point: LocalPoint,
    anchor_time: datetime,
    index: int,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    config: PluginConfig,
) -> _Suspicion:
    raw_point = raw_points[index]
    dt = max((observations[index].event_time - anchor_time).total_seconds(), config.min_dt_seconds)
    distance_to_anchor = _distance(anchor_point, raw_point)
    implied_speed = distance_to_anchor / dt
    hard_jump = (
        implied_speed > config.prefilter_hard_speed_mps
        or (
            dt <= config.prefilter_hard_distance_dt_seconds
            and distance_to_anchor > config.prefilter_hard_distance_m
        )
    )
    bridge_spike = implied_speed > config.prefilter_soft_speed_mps and _is_bridge_spike_from_anchor(
        observations=observations,
        raw_points=raw_points,
        anchor_point=anchor_point,
        anchor_time=anchor_time,
        index=index,
        config=config,
    )
    return _Suspicion(hard_jump=hard_jump, bridge_spike=bridge_spike)


def _is_bridge_spike_from_anchor(
    *,
    observations: list[ParsedMessage],
    raw_points: list[LocalPoint],
    anchor_point: LocalPoint,
    anchor_time: datetime,
    index: int,
    config: PluginConfig,
) -> bool:
    if index >= len(raw_points) - 1:
        return False

    current_point = raw_points[index]
    next_point = raw_points[index + 1]
    neighbor_distance = _distance(anchor_point, next_point)
    current_to_previous = _distance(current_point, anchor_point)
    current_to_next = _distance(current_point, next_point)
    if neighbor_distance > config.prefilter_bridge_neighbor_distance_m:
        return False
    if min(current_to_previous, current_to_next) <= config.prefilter_bridge_center_distance_m:
        return False

    left_dt = max((observations[index].event_time - anchor_time).total_seconds(), config.min_dt_seconds)
    right_dt = _dt_seconds(observations[index], observations[index + 1], config)
    return left_dt <= config.prefilter_hard_distance_dt_seconds and right_dt <= config.prefilter_hard_distance_dt_seconds


def _interpolate_point(
    *,
    left_point: LocalPoint,
    right_point: LocalPoint,
    left_time,
    current_time,
    right_time,
) -> LocalPoint:
    total_seconds = (right_time - left_time).total_seconds()
    if total_seconds <= 0:
        fraction = 0.5
    else:
        fraction = (current_time - left_time).total_seconds() / total_seconds
        fraction = max(0.0, min(1.0, fraction))

    return LocalPoint(
        east_m=left_point.east_m + (right_point.east_m - left_point.east_m) * fraction,
        north_m=left_point.north_m + (right_point.north_m - left_point.north_m) * fraction,
    )


def _extrapolate_point(
    *,
    left_point: LocalPoint,
    right_point: LocalPoint,
    left_time,
    right_time,
    current_time,
    config: PluginConfig,
) -> LocalPoint:
    previous_dt = max((right_time - left_time).total_seconds(), config.min_dt_seconds)
    current_dt = max((current_time - right_time).total_seconds(), config.min_dt_seconds)
    east_velocity = (right_point.east_m - left_point.east_m) / previous_dt
    north_velocity = (right_point.north_m - left_point.north_m) / previous_dt
    return LocalPoint(
        east_m=right_point.east_m + east_velocity * current_dt,
        north_m=right_point.north_m + north_velocity * current_dt,
    )


def _median_filter_points(points: list[LocalPoint], window_size: int) -> list[LocalPoint]:
    if window_size <= 1 or len(points) <= 2:
        return points

    # 中值滤波只负责压普通毛刺，不承担连续 burst 修复职责。
    radius = window_size // 2
    filtered: list[LocalPoint] = []
    for index in range(len(points)):
        start = max(0, index - radius)
        stop = min(len(points), index + radius + 1)
        east_values = [point.east_m for point in points[start:stop]]
        north_values = [point.north_m for point in points[start:stop]]
        filtered.append(LocalPoint(east_m=float(median(east_values)), north_m=float(median(north_values))))
    return filtered


def _distance(left: LocalPoint, right: LocalPoint) -> float:
    east_delta = right.east_m - left.east_m
    north_delta = right.north_m - left.north_m
    return (east_delta * east_delta + north_delta * north_delta) ** 0.5


def _vector(left: LocalPoint, right: LocalPoint) -> tuple[float, float]:
    return right.east_m - left.east_m, right.north_m - left.north_m


def _vector_norm(vector: tuple[float, float]) -> float:
    east_delta, north_delta = vector
    return (east_delta * east_delta + north_delta * north_delta) ** 0.5


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _dt_seconds(previous: ParsedMessage, current: ParsedMessage, config: PluginConfig) -> float:
    return max((current.event_time - previous.event_time).total_seconds(), config.min_dt_seconds)
