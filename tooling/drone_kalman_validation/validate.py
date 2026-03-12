from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .alignment import (
    PairedPoint,
    group_points_by_target,
    haversine_m,
    load_paired_points,
    most_common_non_null,
    reset_output_dir,
    sanitize_filename,
    split_track_segments,
)
from .reporting import save_target_plot


@dataclass(frozen=True, slots=True)
class TargetReportSummary:
    targetId: str | None
    type: int | None
    model: str | None
    message_count: int
    segment_count: int
    output_file: str


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    message_count: int
    target_count: int
    rendered_target_count: int
    targets: list[TargetReportSummary]


@dataclass(slots=True)
class TargetAggregate:
    target_id: str | None
    object_type: int | None
    model: str | None
    points: list[PairedPoint]
    segments: list[list[PairedPoint]]
    raw_total_path_m: float
    smoothed_total_path_m: float

    @property
    def message_count(self) -> int:
        return len(self.points)

    @property
    def segment_count(self) -> int:
        return len(self.segments)


class ValidationError(ValueError):
    pass


def generate_validation_report(
    raw_path: str | Path,
    smoothed_path: str | Path,
    out_dir: str | Path,
    *,
    max_segment_gap_seconds: float = 10.0,
    max_targets: int = 12,
) -> ValidationSummary:
    paired_points = load_paired_points(raw_path, smoothed_path, error_type=ValidationError)
    grouped_points = group_points_by_target(paired_points)
    if not grouped_points:
        raise ValidationError("No aligned targets were found in the input files.")

    aggregates = _build_target_aggregates(grouped_points, max_segment_gap_seconds=max_segment_gap_seconds)
    sorted_aggregates = sorted(aggregates, key=lambda item: (-item.message_count, (item.target_id or "")))

    output_dir = Path(out_dir)
    reset_output_dir(output_dir, subdirs=["targets"], files=["compare.png", "summary.json"])
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    rendered_summaries: list[TargetReportSummary] = []
    for rank, aggregate in enumerate(sorted_aggregates[:max_targets], start=1):
        filename = f"{rank:02d}_{sanitize_filename(aggregate.target_id, default='missing-target-id')}.png"
        save_target_plot(
            target_id=aggregate.target_id,
            model=aggregate.model,
            object_type=aggregate.object_type,
            message_count=aggregate.message_count,
            segment_count=aggregate.segment_count,
            raw_total_path_m=aggregate.raw_total_path_m,
            smoothed_total_path_m=aggregate.smoothed_total_path_m,
            segments=aggregate.segments,
            output_path=targets_dir / filename,
        )
        rendered_summaries.append(
            TargetReportSummary(
                targetId=aggregate.target_id,
                type=aggregate.object_type,
                model=aggregate.model,
                message_count=aggregate.message_count,
                segment_count=aggregate.segment_count,
                output_file=f"targets/{filename}",
            )
        )

    summary = ValidationSummary(
        message_count=len(paired_points),
        target_count=len(sorted_aggregates),
        rendered_target_count=len(rendered_summaries),
        targets=rendered_summaries,
    )
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a simple visual comparison for raw vs smoothed tracks.")
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--smoothed", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-segment-gap-seconds", type=float, default=10.0)
    parser.add_argument("--max-targets", type=int, default=12)
    args = parser.parse_args()

    summary = generate_validation_report(
        raw_path=args.raw,
        smoothed_path=args.smoothed,
        out_dir=args.out_dir,
        max_segment_gap_seconds=args.max_segment_gap_seconds,
        max_targets=args.max_targets,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


def _build_target_aggregates(
    grouped_points: dict[str | None, list[PairedPoint]],
    *,
    max_segment_gap_seconds: float,
) -> list[TargetAggregate]:
    aggregates: list[TargetAggregate] = []
    for target_id, points in grouped_points.items():
        ordered_points = sorted(points, key=lambda item: (item.event_time, item.line_no))
        segments = split_track_segments(ordered_points, max_segment_gap_seconds=max_segment_gap_seconds)
        if not segments:
            continue
        aggregates.append(
            TargetAggregate(
                target_id=target_id,
                object_type=most_common_non_null(point.object_type for point in ordered_points),
                model=most_common_non_null(point.model for point in ordered_points),
                points=ordered_points,
                segments=segments,
                raw_total_path_m=_compute_target_path_length(segments, use_smoothed=False),
                smoothed_total_path_m=_compute_target_path_length(segments, use_smoothed=True),
            )
        )
    return aggregates


def _compute_target_path_length(segments: list[list[PairedPoint]], *, use_smoothed: bool) -> float:
    total = 0.0
    for segment in segments:
        for previous, current in zip(segment, segment[1:]):
            if use_smoothed:
                total += haversine_m(
                    previous.smoothed_latitude,
                    previous.smoothed_longitude,
                    current.smoothed_latitude,
                    current.smoothed_longitude,
                )
            else:
                total += haversine_m(
                    previous.raw_latitude,
                    previous.raw_longitude,
                    current.raw_latitude,
                    current.raw_longitude,
                )
    return round(total, 1)


if __name__ == "__main__":
    raise SystemExit(main())
