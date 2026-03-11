from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .config import PluginConfig
from .message import ParsedMessage, TrackKey, dump_message, parse_message
from .prefilter import RobustPrefilterSegmentSmoother
from .segment import SegmentSmoother


@dataclass(slots=True)
class TrackState:
    # 每条活跃轨迹段的运行时状态。
    segment: SegmentSmoother | RobustPrefilterSegmentSmoother
    last_event_time: datetime
    last_process_time: datetime | None
    trace_id: str | None


class DroneKalmanFilterPlugin:
    def __init__(self, config: PluginConfig | None = None) -> None:
        self.config = config or PluginConfig()
        self._tracks: dict[TrackKey, TrackState] = {}
        # 所有平滑输出先进入待发布队列，再按输入序号统一释放，避免多设备交错时乱序。
        self._pending_outputs: dict[int, dict[str, Any]] = {}
        self._next_input_seq = 1
        self._next_release_seq = 1

    def process(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        parsed = parse_message(self._next_input_seq, message)
        self._next_input_seq += 1

        self._flush_idle_tracks(parsed.arrival_time)

        if parsed.track_key is not None and not parsed.is_smoothable:
            self._flush_track(parsed.track_key)
            self._queue_output(parsed.arrival_seq, parsed.message)
            return self._release_ready()

        if not parsed.is_smoothable:
            self._queue_output(parsed.arrival_seq, parsed.message)
            return self._release_ready()

        state = self._tracks.get(parsed.track_key)
        # traceId 变化、时间倒退或长 gap 都视为新轨迹段，旧段先 flush，再开新段。
        if state is None or self._starts_new_segment(state, parsed):
            self._flush_track(parsed.track_key)
            state = TrackState(
                segment=self._create_segment_smoother(parsed),
                last_event_time=parsed.event_time,
                last_process_time=parsed.arrival_time,
                trace_id=parsed.trace_id,
            )
            self._tracks[parsed.track_key] = state

        for seq, smoothed_message in state.segment.append(parsed):
            self._queue_output(seq, smoothed_message)

        state.last_event_time = parsed.event_time
        state.last_process_time = parsed.arrival_time
        state.trace_id = parsed.trace_id
        return self._release_ready()

    def process_json_line(self, line: str) -> list[str]:
        stripped = line.strip()
        if not stripped:
            return []
        payload = json.loads(stripped)
        return [dump_message(item) for item in self.process(payload)]

    def flush(self) -> list[dict[str, Any]]:
        for key in list(self._tracks):
            self._flush_track(key)
        return self._release_ready()

    def flush_json(self) -> list[str]:
        return [dump_message(item) for item in self.flush()]

    def _starts_new_segment(self, state: TrackState, parsed: ParsedMessage) -> bool:
        if state.trace_id != parsed.trace_id:
            return True
        delta = (parsed.event_time - state.last_event_time).total_seconds()
        if delta <= 0.0:
            return True
        return delta > self.config.max_segment_gap_seconds

    def _flush_idle_tracks(self, current_time: datetime | None) -> None:
        if current_time is None:
            return

        stale_keys: list[TrackKey] = []
        for key, state in self._tracks.items():
            reference_time = state.last_process_time or state.last_event_time
            delta = (current_time - reference_time).total_seconds()
            if delta > self.config.idle_flush_seconds:
                stale_keys.append(key)

        for key in stale_keys:
            self._flush_track(key)

    def _flush_track(self, key: TrackKey) -> None:
        state = self._tracks.pop(key, None)
        if state is None:
            return
        for seq, message in state.segment.flush():
            self._queue_output(seq, message)

    def _create_segment_smoother(self, parsed: ParsedMessage) -> SegmentSmoother | RobustPrefilterSegmentSmoother:
        kwargs = {
            "trace_id": parsed.trace_id,
            "config": self.config,
            "anchor_latitude": parsed.latitude,
            "anchor_longitude": parsed.longitude,
        }
        if self.config.smoother_mode == "robust_prefilter_kalman":
            return RobustPrefilterSegmentSmoother(**kwargs)
        return SegmentSmoother(**kwargs)

    def _queue_output(self, arrival_seq: int, message: dict[str, Any]) -> None:
        if arrival_seq in self._pending_outputs:
            raise ValueError(f"duplicate output for arrival_seq={arrival_seq}")
        self._pending_outputs[arrival_seq] = message

    def _release_ready(self) -> list[dict[str, Any]]:
        released: list[dict[str, Any]] = []
        # 只有最早未释放的序号已经成熟，才允许真正对外输出。
        while self._next_release_seq in self._pending_outputs:
            released.append(self._pending_outputs.pop(self._next_release_seq))
            self._next_release_seq += 1
        return released
