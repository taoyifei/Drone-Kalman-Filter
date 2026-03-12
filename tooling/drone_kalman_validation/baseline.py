from __future__ import annotations

import argparse
import json
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drone_kalman_filter.cli import _DEFAULT_CONFIG
from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.geo import LocalTangentPlane
from drone_kalman_filter.kalman import smooth_positions
from drone_kalman_filter.message import ParsedMessage, TrackKey, dump_message, parse_message, set_position_strings


@dataclass(slots=True)
class OfflineSegmentState:
    trace_id: str | None
    last_event_time: object
    messages: list[ParsedMessage]


def generate_offline_rts_messages(
    messages: list[dict[str, Any]],
    config: PluginConfig | None = None,
) -> list[dict[str, Any]]:
    effective_config = _baseline_config(config)
    track_states: dict[TrackKey, OfflineSegmentState] = {}
    outputs: dict[int, dict[str, Any]] = {}

    for index, message in enumerate(messages, start=1):
        parsed = parse_message(index, message)
        if parsed.track_key is not None and not parsed.is_smoothable:
            _flush_track(track_states, outputs, parsed.track_key, effective_config)
            outputs[parsed.arrival_seq] = parsed.message
            continue

        if not parsed.is_smoothable:
            outputs[parsed.arrival_seq] = parsed.message
            continue

        state = track_states.get(parsed.track_key)
        if state is None or _starts_new_segment(state, parsed, effective_config):
            _flush_track(track_states, outputs, parsed.track_key, effective_config)
            state = OfflineSegmentState(trace_id=parsed.trace_id, last_event_time=parsed.event_time, messages=[])
            track_states[parsed.track_key] = state

        state.messages.append(parsed)
        state.last_event_time = parsed.event_time
        state.trace_id = parsed.trace_id

    for track_key in list(track_states):
        _flush_track(track_states, outputs, track_key, effective_config)

    return [outputs[index] for index in range(1, len(messages) + 1)]


def generate_offline_rts_json_lines(
    raw_lines: list[str],
    config: PluginConfig | None = None,
) -> list[str]:
    messages = [json.loads(line) for line in raw_lines if line.strip()]
    outputs = generate_offline_rts_messages(messages, config=config)
    return [dump_message(message) for message in outputs]


def write_offline_rts_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    config: PluginConfig | None = None,
) -> None:
    lines = Path(input_path).read_text(encoding="utf-8").splitlines()
    output_lines = generate_offline_rts_json_lines(lines, config=config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an offline fixed-interval RTS baseline JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-dt-seconds", type=float, default=_DEFAULT_CONFIG.min_dt_seconds)
    parser.add_argument("--max-segment-gap-seconds", type=float, default=_DEFAULT_CONFIG.max_segment_gap_seconds)
    parser.add_argument("--measurement-sigma-m", type=float, default=_DEFAULT_CONFIG.measurement_sigma_m)
    parser.add_argument("--process-accel-sigma-mps2", type=float, default=_DEFAULT_CONFIG.process_accel_sigma_mps2)
    parser.add_argument("--initial-position-sigma-m", type=float, default=_DEFAULT_CONFIG.initial_position_sigma_m)
    parser.add_argument("--initial-velocity-sigma-mps", type=float, default=_DEFAULT_CONFIG.initial_velocity_sigma_mps)
    parser.add_argument("--soft-residual-speed-mps", type=float, default=_DEFAULT_CONFIG.soft_residual_speed_mps)
    parser.add_argument("--hard-residual-speed-mps", type=float, default=_DEFAULT_CONFIG.hard_residual_speed_mps)
    parser.add_argument("--soft-noise-scale", type=float, default=_DEFAULT_CONFIG.soft_noise_scale)
    args = parser.parse_args()

    config = PluginConfig(
        window_size=_DEFAULT_CONFIG.window_size,
        lag_points=_DEFAULT_CONFIG.lag_points,
        min_dt_seconds=args.min_dt_seconds,
        max_segment_gap_seconds=args.max_segment_gap_seconds,
        idle_flush_seconds=_DEFAULT_CONFIG.idle_flush_seconds,
        measurement_sigma_m=args.measurement_sigma_m,
        process_accel_sigma_mps2=args.process_accel_sigma_mps2,
        initial_position_sigma_m=args.initial_position_sigma_m,
        initial_velocity_sigma_mps=args.initial_velocity_sigma_mps,
        soft_residual_speed_mps=args.soft_residual_speed_mps,
        hard_residual_speed_mps=args.hard_residual_speed_mps,
        soft_noise_scale=args.soft_noise_scale,
        reset_velocity_on_reject=True,
    )
    write_offline_rts_jsonl(args.input, args.output, config=config)
    return 0


def _starts_new_segment(state: OfflineSegmentState, parsed: ParsedMessage, config: PluginConfig) -> bool:
    if state.trace_id != parsed.trace_id:
        return True
    delta = (parsed.event_time - state.last_event_time).total_seconds()
    if delta <= 0.0:
        return True
    return delta > config.max_segment_gap_seconds


def _flush_track(
    track_states: dict[TrackKey, OfflineSegmentState],
    outputs: dict[int, dict[str, Any]],
    track_key: TrackKey,
    config: PluginConfig,
) -> None:
    state = track_states.pop(track_key, None)
    if state is None or not state.messages:
        return

    first = state.messages[0]
    plane = LocalTangentPlane(first.latitude, first.longitude)
    smoothed_positions = smooth_positions(state.messages, plane, config)
    for parsed, smoothed_position in zip(state.messages, smoothed_positions):
        latitude, longitude = plane.to_geodetic(smoothed_position.east_m, smoothed_position.north_m)
        outputs[parsed.arrival_seq] = set_position_strings(parsed, latitude, longitude)


def _baseline_config(config: PluginConfig | None) -> PluginConfig:
    if config is None:
        return PluginConfig(reset_velocity_on_reject=True)
    if config.reset_velocity_on_reject:
        return config
    return replace(config, reset_velocity_on_reject=True)


if __name__ == "__main__":
    raise SystemExit(main())
