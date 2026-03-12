from __future__ import annotations

from pathlib import Path
from typing import Sequence
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from drone_kalman_filter.geo import LocalTangentPlane

from .alignment import PairedPoint, TriplePoint


def save_target_plot(
    *,
    target_id: str | None,
    model: str | None,
    object_type: int | None,
    message_count: int,
    segment_count: int,
    raw_total_path_m: float,
    smoothed_total_path_m: float,
    segments: Sequence[Sequence[PairedPoint]],
    output_path: str | Path,
) -> None:
    first_point = segments[0][0]
    last_point = segments[-1][-1]
    plane = LocalTangentPlane(first_point.raw_latitude, first_point.raw_longitude)

    figure, axis = plt.subplots(figsize=(12, 10), constrained_layout=True)
    axis.set_facecolor("#fbfcfe")

    for index, segment in enumerate(segments):
        raw_local = [plane.to_local(point.raw_latitude, point.raw_longitude) for point in segment]
        smoothed_local = [plane.to_local(point.smoothed_latitude, point.smoothed_longitude) for point in segment]
        raw_east = [value[0] for value in raw_local]
        raw_north = [value[1] for value in raw_local]
        smoothed_east = [value[0] for value in smoothed_local]
        smoothed_north = [value[1] for value in smoothed_local]

        axis.plot(
            raw_east,
            raw_north,
            color="#6b7280",
            linewidth=1.0,
            alpha=0.50,
            linestyle="--",
            label="raw" if index == 0 else None,
            zorder=1,
        )
        axis.plot(
            smoothed_east,
            smoothed_north,
            color="#1565c0",
            linewidth=2.4,
            alpha=0.95,
            label="smoothed" if index == 0 else None,
            zorder=2,
        )
        if len(segment) == 1:
            axis.scatter(raw_east, raw_north, color="#9aa0a6", s=14, alpha=0.45, zorder=1)
            axis.scatter(smoothed_east, smoothed_north, color="#1565c0", s=18, alpha=0.95, zorder=2)

    start_raw = plane.to_local(first_point.raw_latitude, first_point.raw_longitude)
    end_raw = plane.to_local(last_point.raw_latitude, last_point.raw_longitude)
    start_smoothed = plane.to_local(first_point.smoothed_latitude, first_point.smoothed_longitude)
    end_smoothed = plane.to_local(last_point.smoothed_latitude, last_point.smoothed_longitude)

    axis.scatter([start_raw[0]], [start_raw[1]], color="#c7cacf", s=40, marker="o", alpha=0.8, zorder=3)
    axis.scatter([end_raw[0]], [end_raw[1]], color="#c7cacf", s=55, marker="X", alpha=0.8, zorder=3)
    axis.scatter([start_smoothed[0]], [start_smoothed[1]], color="#1b9e5a", s=90, marker="o", zorder=4)
    axis.scatter([end_smoothed[0]], [end_smoothed[1]], color="#d93025", s=110, marker="X", zorder=4)
    axis.annotate("Start", xy=start_smoothed, xytext=(8, 8), textcoords="offset points", color="#1b5e20", fontsize=10)
    axis.annotate("End", xy=end_smoothed, xytext=(8, -14), textcoords="offset points", color="#8b1e16", fontsize=10)

    title_parts = [f"targetId: {target_id or '(missing)'}"]
    if model:
        title_parts.append(model)
    if object_type is not None:
        title_parts.append(f"type={object_type}")
    title_parts.append(f"{message_count} points")
    axis.set_title(" | ".join(title_parts), fontsize=15, pad=16)
    axis.set_xlabel("East (m)")
    axis.set_ylabel("North (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.12)
    axis.legend(loc="upper right", frameon=True)

    info_text = (
        f"segments: {segment_count}\n"
        f"raw path: {raw_total_path_m:,.1f} m\n"
        f"smoothed path: {smoothed_total_path_m:,.1f} m"
    )
    axis.text(
        0.98,
        0.02,
        info_text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.95},
    )

    _save_figure(figure, output_path, dpi=170)


def save_diagnostic_plot(
    *,
    segment: Sequence[TriplePoint],
    current_offsets: Sequence[float],
    raw_offsets: Sequence[float],
    output_path: str | Path,
) -> None:
    first = segment[0]
    plane = LocalTangentPlane(first.raw_latitude, first.raw_longitude)
    raw_local = [plane.to_local(point.raw_latitude, point.raw_longitude) for point in segment]
    current_local = [plane.to_local(point.current_latitude, point.current_longitude) for point in segment]
    baseline_local = [plane.to_local(point.baseline_latitude, point.baseline_longitude) for point in segment]
    times = [(point.event_time - first.event_time).total_seconds() for point in segment]

    raw_xy = _split_xy(raw_local)
    current_xy = _split_xy(current_local)
    baseline_xy = _split_xy(baseline_local)
    zoom_bounds = _combine_bounds(
        _compute_bounds(raw_local),
        _compute_bounds(current_local),
    )
    full_bounds = _combine_bounds(
        _compute_bounds(raw_local),
        _compute_bounds(current_local),
        _compute_bounds(baseline_local),
    )

    figure = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.2, 0.9])
    overview_axis = figure.add_subplot(grid[0, 0])
    zoom_axis = figure.add_subplot(grid[0, 1])
    offset_axis = figure.add_subplot(grid[1, :])

    _plot_spatial_series(overview_axis, raw_xy, current_xy, baseline_xy)
    overview_axis.set_title(
        "Overview: raw / current / baseline",
        fontsize=12,
    )
    overview_axis.set_xlabel("East (m)")
    overview_axis.set_ylabel("North (m)")
    overview_axis.set_xlim(full_bounds[0], full_bounds[1])
    overview_axis.set_ylim(full_bounds[2], full_bounds[3])
    overview_axis.set_aspect("equal", adjustable="box")
    overview_axis.grid(True, alpha=0.15)
    overview_axis.legend(loc="upper right")
    overview_axis.add_patch(
        plt.Rectangle(
            (zoom_bounds[0], zoom_bounds[2]),
            zoom_bounds[1] - zoom_bounds[0],
            zoom_bounds[3] - zoom_bounds[2],
            fill=False,
            edgecolor="#b3261e",
            linewidth=1.4,
            linestyle=":",
            zorder=5,
        )
    )

    zoom_baseline_xy = _clip_xy_to_bounds(baseline_xy, zoom_bounds)
    _plot_spatial_series(zoom_axis, raw_xy, current_xy, zoom_baseline_xy)
    zoom_axis.set_title(
        "Detail: zoom to raw/current extent",
        fontsize=12,
    )
    zoom_axis.set_xlabel("East (m)")
    zoom_axis.set_ylabel("North (m)")
    zoom_axis.set_xlim(zoom_bounds[0], zoom_bounds[1])
    zoom_axis.set_ylim(zoom_bounds[2], zoom_bounds[3])
    zoom_axis.set_aspect("auto")
    zoom_axis.grid(True, alpha=0.15)
    zoom_axis.text(
        0.02,
        0.02,
        "Baseline may leave this zoomed view if it drifts far away.",
        transform=zoom_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.92},
    )

    offset_axis.plot(times, raw_offsets, color="#7a7a7a", linestyle="--", alpha=0.65, label="raw vs baseline")
    offset_axis.plot(times, current_offsets, color="#d93025", linewidth=2.0, label="current vs baseline")
    offset_axis.set_title("Offset to fixed-interval RTS baseline", fontsize=12)
    offset_axis.set_xlabel("Time since segment start (s)")
    offset_axis.set_ylabel("Offset (m)")
    offset_axis.grid(True, alpha=0.15)
    offset_axis.legend(loc="upper right")

    figure.suptitle(
        f"targetId: {first.target_id} | deviceId: {first.device_id} | traceId: {first.trace_id} | "
        f"{len(segment)} points",
        fontsize=14,
    )

    _save_figure(figure, output_path, dpi=160)


def _plot_spatial_series(
    axis: plt.Axes,
    raw_xy: tuple[list[float], list[float]],
    current_xy: tuple[list[float], list[float]],
    baseline_xy: tuple[list[float], list[float]],
) -> None:
    axis.plot(
        raw_xy[0],
        raw_xy[1],
        color="#7a7a7a",
        linestyle="--",
        alpha=0.55,
        label="raw",
    )
    axis.plot(
        current_xy[0],
        current_xy[1],
        color="#1565c0",
        linewidth=2.0,
        label="current",
    )
    axis.plot(
        baseline_xy[0],
        baseline_xy[1],
        color="#1b9e5a",
        linewidth=2.0,
        label="baseline RTS",
    )


def _split_xy(local_points: Sequence[tuple[float, float]]) -> tuple[list[float], list[float]]:
    return [value[0] for value in local_points], [value[1] for value in local_points]


def _clip_xy_to_bounds(
    xy: tuple[Sequence[float], Sequence[float]],
    bounds: tuple[float, float, float, float],
) -> tuple[list[float], list[float]]:
    clipped_x: list[float] = []
    clipped_y: list[float] = []
    min_x, max_x, min_y, max_y = bounds
    for x_value, y_value in zip(xy[0], xy[1]):
        if min_x <= x_value <= max_x and min_y <= y_value <= max_y:
            clipped_x.append(float(x_value))
            clipped_y.append(float(y_value))
            continue
        clipped_x.append(math.nan)
        clipped_y.append(math.nan)
    return clipped_x, clipped_y


def _compute_bounds(
    local_points: Sequence[tuple[float, float]],
    *,
    padding_ratio: float = 0.08,
    minimum_span_m: float = 20.0,
) -> tuple[float, float, float, float]:
    east_values = [value[0] for value in local_points]
    north_values = [value[1] for value in local_points]
    east_min, east_max = min(east_values), max(east_values)
    north_min, north_max = min(north_values), max(north_values)
    east_min, east_max = _expand_span(east_min, east_max, minimum_span_m)
    north_min, north_max = _expand_span(north_min, north_max, minimum_span_m)
    east_pad = max((east_max - east_min) * padding_ratio, 5.0)
    north_pad = max((north_max - north_min) * padding_ratio, 5.0)
    return (
        east_min - east_pad,
        east_max + east_pad,
        north_min - north_pad,
        north_max + north_pad,
    )


def _combine_bounds(*bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (
        min(bound[0] for bound in bounds),
        max(bound[1] for bound in bounds),
        min(bound[2] for bound in bounds),
        max(bound[3] for bound in bounds),
    )


def _expand_span(lower: float, upper: float, minimum_span_m: float) -> tuple[float, float]:
    span = upper - lower
    if span >= minimum_span_m:
        return lower, upper
    midpoint = (lower + upper) / 2.0
    half_span = minimum_span_m / 2.0
    return midpoint - half_span, midpoint + half_span


def _save_figure(figure: plt.Figure, output_path: str | Path, *, dpi: int) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
