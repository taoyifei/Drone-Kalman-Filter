from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .io import (
    group_points_by_device,
    load_compare_points,
    reset_output_dir,
    sanitize_filename,
    split_track_segments,
)
from .plotting import save_device_compare_plot


@dataclass(frozen=True, slots=True)
class DeviceCompareSummary:
    targetId: str | None
    deviceId: str | None
    total_message_count: int
    total_segment_count: int
    selected_segment_point_count: int
    selected_segment_start_time: str
    selected_segment_end_time: str
    output_file: str


@dataclass(frozen=True, slots=True)
class CompareSummary:
    message_count: int
    device_count: int
    devices: list[DeviceCompareSummary]


class CompareError(ValueError):
    pass


def generate_compare_report(
    raw_path: str | Path,
    smoothed_path: str | Path,
    out_dir: str | Path,
    *,
    max_segment_gap_seconds: float = 10.0,
) -> CompareSummary:
    compare_points = load_compare_points(raw_path, smoothed_path, error_type=CompareError)
    grouped_points = group_points_by_device(compare_points)
    if not grouped_points:
        raise CompareError("No aligned device tracks were found in the input files.")

    devices = []
    for key, points in grouped_points.items():
        target_id, device_id = key
        ordered_points = sorted(points, key=lambda item: (item.event_time, item.line_no))
        segments = split_track_segments(ordered_points, max_segment_gap_seconds=max_segment_gap_seconds)
        if not segments:
            continue
        selected_segment = min(
            segments,
            key=lambda segment: (
                -len(segment),
                segment[0].event_time,
                segment[0].line_no,
            ),
        )
        devices.append((target_id, device_id, ordered_points, segments, selected_segment))

    devices.sort(key=lambda item: (-len(item[2]), item[0] or "", item[1] or ""))

    output_dir = Path(out_dir)
    reset_output_dir(output_dir, subdirs=["devices"], files=["summary.json"])
    devices_dir = output_dir / "devices"
    devices_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[DeviceCompareSummary] = []
    for rank, (target_id, device_id, points, segments, selected_segment) in enumerate(devices, start=1):
        filename = (
            f"{rank:02d}_{sanitize_filename(target_id, default='missing-target-id')}"
            f"__{sanitize_filename(device_id, default='missing-device-id')}.png"
        )
        save_device_compare_plot(
            target_id=target_id,
            device_id=device_id,
            selected_segment_point_count=len(selected_segment),
            total_segment_count=len(segments),
            segment=selected_segment,
            output_path=devices_dir / filename,
        )
        summaries.append(
            DeviceCompareSummary(
                targetId=target_id,
                deviceId=device_id,
                total_message_count=len(points),
                total_segment_count=len(segments),
                selected_segment_point_count=len(selected_segment),
                selected_segment_start_time=_format_datetime(selected_segment[0].event_time),
                selected_segment_end_time=_format_datetime(selected_segment[-1].event_time),
                output_file=f"devices/{filename}",
            )
        )

    summary = CompareSummary(
        message_count=len(compare_points),
        device_count=len(summaries),
        devices=summaries,
    )
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simple raw vs smoothed track comparison plots by device.")
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--smoothed", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-segment-gap-seconds", type=float, default=10.0)
    args = parser.parse_args()

    summary = generate_compare_report(
        raw_path=args.raw,
        smoothed_path=args.smoothed,
        out_dir=args.out_dir,
        max_segment_gap_seconds=args.max_segment_gap_seconds,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


def _format_datetime(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
