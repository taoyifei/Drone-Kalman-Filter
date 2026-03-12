from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from drone_kalman_filter.config import PluginConfig

from .alignment import (
    TriplePoint,
    build_offset_stats,
    build_triple_points,
    haversine_m,
    load_jsonl_lines,
    median,
    reset_output_dir,
    sanitize_filename,
    split_track_segments,
)
from .baseline import generate_offline_rts_json_lines
from .reporting import save_diagnostic_plot


@dataclass(frozen=True, slots=True)
class DiagnosticSegmentSummary:
    rank: int
    targetId: str | None
    deviceId: str | None
    traceId: str | None
    start_event_time: str | None
    end_event_time: str | None
    point_count: int
    current_vs_baseline_max_offset_m: float
    current_vs_baseline_median_offset_m: float
    raw_vs_baseline_max_offset_m: float
    raw_start_to_baseline_start_m: float
    raw_end_to_baseline_end_m: float
    current_start_to_baseline_start_m: float
    raw_path_m: float
    current_path_m: float
    baseline_path_m: float
    output_file: str
    points_file: str


@dataclass(frozen=True, slots=True)
class DiagnosticSummary:
    baseline_generation_mode: str
    message_count: int
    segment_count: int
    current_vs_baseline_offset_m: dict[str, float | None]
    raw_vs_baseline_offset_m: dict[str, float | None]
    top_segments: list[DiagnosticSegmentSummary]


class DiagnosticError(ValueError):
    pass


def generate_diagnostic_report(
    raw_path: str | Path,
    smoothed_path: str | Path,
    out_dir: str | Path,
    *,
    config: PluginConfig | None = None,
    baseline_output_path: str | Path | None = None,
    max_segments: int = 8,
    min_segment_points: int = 5,
) -> DiagnosticSummary:
    effective_config = config or PluginConfig()
    raw_lines = load_jsonl_lines(raw_path)
    current_lines = load_jsonl_lines(smoothed_path)
    baseline_lines = generate_offline_rts_json_lines(raw_lines, config=effective_config)

    if baseline_output_path is not None:
        baseline_path = Path(baseline_output_path)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")

    triple_points = build_triple_points(raw_lines, current_lines, baseline_lines, error_type=DiagnosticError)
    segments = _split_segments_for_diagnostics(
        triple_points,
        max_segment_gap_seconds=effective_config.max_segment_gap_seconds,
    )
    if not segments:
        raise DiagnosticError("No diagnostic segments were found.")

    ranked_segments = _rank_segments(segments, min_segment_points=min_segment_points)
    output_dir = Path(out_dir)
    reset_output_dir(output_dir, subdirs=["segments"], files=["summary.json"])
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[DiagnosticSegmentSummary] = []
    for rank, segment in enumerate(ranked_segments[:max_segments], start=1):
        stem = _build_segment_stem(rank, segment)
        image_filename = f"{stem}.png"
        points_filename = f"{stem}.json"
        current_offsets = [_offset_current_vs_baseline(point) for point in segment]
        raw_offsets = [_offset_raw_vs_baseline(point) for point in segment]
        save_diagnostic_plot(
            segment=segment,
            current_offsets=current_offsets,
            raw_offsets=raw_offsets,
            output_path=segments_dir / image_filename,
        )
        _write_segment_points_json(
            output_path=segments_dir / points_filename,
            segment=segment,
            current_offsets=current_offsets,
            raw_offsets=raw_offsets,
        )
        summaries.append(
            _build_segment_summary(
                rank=rank,
                segment=segment,
                image_filename=image_filename,
                points_filename=points_filename,
                current_offsets=current_offsets,
                raw_offsets=raw_offsets,
            )
        )

    summary = DiagnosticSummary(
        baseline_generation_mode="fixed_interval_rts",
        message_count=len(triple_points),
        segment_count=len(segments),
        current_vs_baseline_offset_m=build_offset_stats(
            [_offset_current_vs_baseline(point) for point in triple_points if point.has_valid_coordinates]
        ),
        raw_vs_baseline_offset_m=build_offset_stats(
            [_offset_raw_vs_baseline(point) for point in triple_points if point.has_valid_coordinates]
        ),
        top_segments=summaries,
    )
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _rank_segments(segments: list[list[TriplePoint]], *, min_segment_points: int) -> list[list[TriplePoint]]:
    eligible = [segment for segment in segments if len(segment) >= min_segment_points]
    if not eligible:
        eligible = segments
    return sorted(eligible, key=lambda segment: (-max(_offset_current_vs_baseline(point) for point in segment), -len(segment)))


def _split_segments_for_diagnostics(
    points: list[TriplePoint],
    *,
    max_segment_gap_seconds: float,
) -> list[list[TriplePoint]]:
    grouped: dict[tuple[str | None, str | None], list[TriplePoint]] = {}
    for point in points:
        key = (point.target_id, point.device_id)
        grouped.setdefault(key, []).append(point)

    segments: list[list[TriplePoint]] = []
    for group_points in grouped.values():
        ordered_points = sorted(group_points, key=lambda item: item.line_no)
        segments.extend(split_track_segments(ordered_points, max_segment_gap_seconds=max_segment_gap_seconds))

    return sorted(segments, key=lambda segment: segment[0].line_no)


def _build_segment_summary(
    rank: int,
    segment: list[TriplePoint],
    image_filename: str,
    points_filename: str,
    current_offsets: list[float],
    raw_offsets: list[float],
) -> DiagnosticSegmentSummary:
    raw_path, current_path, baseline_path = _compute_paths(segment)
    first = segment[0]
    last = segment[-1]
    return DiagnosticSegmentSummary(
        rank=rank,
        targetId=first.target_id,
        deviceId=first.device_id,
        traceId=first.trace_id,
        start_event_time=first.event_time_text,
        end_event_time=last.event_time_text,
        point_count=len(segment),
        current_vs_baseline_max_offset_m=round(max(current_offsets), 4),
        current_vs_baseline_median_offset_m=round(median(current_offsets), 4),
        raw_vs_baseline_max_offset_m=round(max(raw_offsets), 4),
        raw_start_to_baseline_start_m=round(_offset_raw_vs_baseline(first), 4),
        raw_end_to_baseline_end_m=round(_offset_raw_vs_baseline(last), 4),
        current_start_to_baseline_start_m=round(_offset_current_vs_baseline(first), 4),
        raw_path_m=round(raw_path, 4),
        current_path_m=round(current_path, 4),
        baseline_path_m=round(baseline_path, 4),
        output_file=f"segments/{image_filename}",
        points_file=f"segments/{points_filename}",
    )


def _compute_paths(segment: list[TriplePoint]) -> tuple[float, float, float]:
    raw_path = 0.0
    current_path = 0.0
    baseline_path = 0.0
    for previous, current in zip(segment, segment[1:]):
        raw_path += haversine_m(previous.raw_latitude, previous.raw_longitude, current.raw_latitude, current.raw_longitude)
        current_path += haversine_m(
            previous.current_latitude,
            previous.current_longitude,
            current.current_latitude,
            current.current_longitude,
        )
        baseline_path += haversine_m(
            previous.baseline_latitude,
            previous.baseline_longitude,
            current.baseline_latitude,
            current.baseline_longitude,
        )
    return raw_path, current_path, baseline_path


def _offset_current_vs_baseline(point: TriplePoint) -> float:
    return haversine_m(
        point.current_latitude,
        point.current_longitude,
        point.baseline_latitude,
        point.baseline_longitude,
    )


def _offset_raw_vs_baseline(point: TriplePoint) -> float:
    return haversine_m(
        point.raw_latitude,
        point.raw_longitude,
        point.baseline_latitude,
        point.baseline_longitude,
    )


def _offset_raw_vs_current(point: TriplePoint) -> float:
    return haversine_m(
        point.raw_latitude,
        point.raw_longitude,
        point.current_latitude,
        point.current_longitude,
    )


def _build_segment_stem(rank: int, segment: list[TriplePoint]) -> str:
    first = segment[0]
    target = sanitize_filename(first.target_id, default="missing")
    device = sanitize_filename(first.device_id, default="missing")
    trace = sanitize_filename(first.trace_id, default="missing")
    return f"{rank:02d}_{target}_{device}_{trace}_{len(segment)}pts"


def _write_segment_points_json(
    *,
    output_path: str | Path,
    segment: list[TriplePoint],
    current_offsets: list[float],
    raw_offsets: list[float],
) -> None:
    payload = {
        "targetId": segment[0].target_id,
        "deviceId": segment[0].device_id,
        "traceId": segment[0].trace_id,
        "start_event_time": segment[0].event_time_text,
        "end_event_time": segment[-1].event_time_text,
        "point_count": len(segment),
        "points": [
            {
                "line_no": point.line_no,
                "targetId": point.target_id,
                "deviceId": point.device_id,
                "traceId": point.trace_id,
                "eventTime": point.event_time_text,
                "raw": {
                    "latitude": point.raw_latitude,
                    "longitude": point.raw_longitude,
                },
                "current": {
                    "latitude": point.current_latitude,
                    "longitude": point.current_longitude,
                },
                "baseline": {
                    "latitude": point.baseline_latitude,
                    "longitude": point.baseline_longitude,
                },
                "raw_vs_baseline_offset_m": round(raw_offsets[index], 4),
                "current_vs_baseline_offset_m": round(current_offsets[index], 4),
                "raw_vs_current_offset_m": round(_offset_raw_vs_current(point), 4),
            }
            for index, point in enumerate(segment)
        ],
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
