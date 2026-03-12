from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .io import ComparePoint


def save_device_compare_plot(
    *,
    target_id: str | None,
    device_id: str | None,
    selected_segment_point_count: int,
    total_segment_count: int,
    segment: Sequence[ComparePoint],
    output_path: str | Path,
) -> None:
    raw_points = [(point.raw_longitude, point.raw_latitude) for point in segment]
    smoothed_points = [(point.smoothed_longitude, point.smoothed_latitude) for point in segment]
    bounds = _combine_bounds(_compute_bounds(raw_points), _compute_bounds(smoothed_points))

    figure, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    raw_axis, smoothed_axis = axes
    for axis in axes:
        axis.set_facecolor("#fbfcfe")

    raw_xy = _split_xy([(point.raw_longitude, point.raw_latitude) for point in segment])
    smoothed_xy = _split_xy([(point.smoothed_longitude, point.smoothed_latitude) for point in segment])
    _plot_track(
        raw_axis,
        raw_xy,
        color="#6b7280",
        linewidth=1.8,
        alpha=0.90,
        label="raw",
    )
    _plot_track(
        smoothed_axis,
        smoothed_xy,
        color="#1565c0",
        linewidth=2.0,
        alpha=0.95,
        label="smoothed",
    )

    _annotate_endpoints(
        raw_axis,
        (segment[0].raw_longitude, segment[0].raw_latitude),
        (segment[-1].raw_longitude, segment[-1].raw_latitude),
    )
    _annotate_endpoints(
        smoothed_axis,
        (segment[0].smoothed_longitude, segment[0].smoothed_latitude),
        (segment[-1].smoothed_longitude, segment[-1].smoothed_latitude),
    )

    title = (
        f"targetId: {target_id or '(missing)'} | deviceId: {device_id or '(missing)'} | "
        f"{selected_segment_point_count} points"
    )
    figure.suptitle(title, fontsize=15)

    _configure_axis(raw_axis, "Raw Track", bounds)
    _configure_axis(smoothed_axis, "Smoothed Track", bounds)
    raw_axis.legend(loc="upper right", frameon=True)
    smoothed_axis.legend(loc="upper right", frameon=True)

    info_text = f"total segments: {total_segment_count}"
    smoothed_axis.text(
        0.98,
        0.02,
        info_text,
        transform=smoothed_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.95},
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_track(
    axis: plt.Axes,
    xy: tuple[list[float], list[float]],
    *,
    color: str,
    linewidth: float,
    alpha: float,
    label: str | None,
) -> None:
    axis.plot(
        xy[0],
        xy[1],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )
    if len(xy[0]) == 1:
        axis.scatter(xy[0], xy[1], color=color, s=18, alpha=alpha)


def _annotate_endpoints(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.scatter([start[0]], [start[1]], color="#1b9e5a", s=44, marker="o", zorder=4)
    axis.scatter([end[0]], [end[1]], color="#d93025", s=56, marker="X", zorder=4)


def _configure_axis(
    axis: plt.Axes,
    title: str,
    bounds: tuple[float, float, float, float],
) -> None:
    axis.set_title(title, fontsize=12)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_ylim(bounds[2], bounds[3])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.12)


def _split_xy(points: Sequence[tuple[float, float]]) -> tuple[list[float], list[float]]:
    return [point[0] for point in points], [point[1] for point in points]


def _compute_bounds(
    points: Sequence[tuple[float, float]],
    *,
    padding_ratio: float = 0.08,
    minimum_span: float = 0.0002,
) -> tuple[float, float, float, float]:
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_min, x_max = _expand_span(x_min, x_max, minimum_span)
    y_min, y_max = _expand_span(y_min, y_max, minimum_span)
    x_pad = max((x_max - x_min) * padding_ratio, minimum_span / 4.0)
    y_pad = max((y_max - y_min) * padding_ratio, minimum_span / 4.0)
    return (
        x_min - x_pad,
        x_max + x_pad,
        y_min - y_pad,
        y_max + y_pad,
    )


def _combine_bounds(*bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (
        min(bound[0] for bound in bounds),
        max(bound[1] for bound in bounds),
        min(bound[2] for bound in bounds),
        max(bound[3] for bound in bounds),
    )


def _expand_span(lower: float, upper: float, minimum_span: float) -> tuple[float, float]:
    span = upper - lower
    if span >= minimum_span:
        return lower, upper
    midpoint = (lower + upper) / 2.0
    half_span = minimum_span / 2.0
    return midpoint - half_span, midpoint + half_span
