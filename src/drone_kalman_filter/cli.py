"""主链路命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drone_kalman_filter.config import PluginConfig
from drone_kalman_filter.metrics import build_report, write_acceptance_summary
from drone_kalman_filter.plugin import DroneKalmanFilterPlugin

_DEFAULT_CONFIG = PluginConfig()


def main() -> int:
    """解析命令行参数并分发到对应子命令。

    Args:
        None. 不接收额外参数。

    Returns:
        int: 进程退出码。
    """
    parser = argparse.ArgumentParser(
        description="Kalman smoother for DetectionTarget JSONL streams.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smooth_parser = subparsers.add_parser("smooth", help="Smooth a JSONL file.")
    smooth_parser.add_argument("--input", required=True, type=Path)
    smooth_parser.add_argument("--output", required=True, type=Path)
    _add_config_arguments(smooth_parser)

    report_parser = subparsers.add_parser(
        "report", help="Print movement statistics for a JSONL file.")
    report_parser.add_argument("--input", required=True, type=Path)

    acceptance_parser = subparsers.add_parser(
        "acceptance",
        help="Generate acceptance metrics for raw vs smoothed JSONL.")
    acceptance_parser.add_argument("--raw", required=True, type=Path)
    acceptance_parser.add_argument("--smoothed", required=True, type=Path)
    acceptance_parser.add_argument("--output", required=True, type=Path)
    acceptance_parser.add_argument("--lag-points",
                                   type=int,
                                   default=_DEFAULT_CONFIG.lag_points)
    acceptance_parser.add_argument("--min-dt-seconds",
                                   type=float,
                                   default=_DEFAULT_CONFIG.min_dt_seconds)
    acceptance_parser.add_argument(
        "--max-segment-gap-seconds",
        type=float,
        default=_DEFAULT_CONFIG.max_segment_gap_seconds)
    acceptance_parser.add_argument("--idle-flush-seconds",
                                   type=float,
                                   default=_DEFAULT_CONFIG.idle_flush_seconds)

    args = parser.parse_args()
    if args.command == "smooth":
        return _run_smooth(args)
    if args.command == "acceptance":
        return _run_acceptance(args)
    return _run_report(args)


def _run_smooth(args: argparse.Namespace) -> int:
    """按给定配置平滑输入 JSONL 并写出结果。

    Args:
        args: 命令行解析后的参数对象。

    Returns:
        int: 进程退出码。
    """
    # smooth 命令直接把 JSONL 逐行喂给实时插件，因此和真实接流的行为是一致的。
    config = PluginConfig(
        smoother_mode=args.smoother_mode,
        window_size=args.window_size,
        lag_points=args.lag_points,
        min_dt_seconds=args.min_dt_seconds,
        max_segment_gap_seconds=args.max_segment_gap_seconds,
        idle_flush_seconds=args.idle_flush_seconds,
        measurement_sigma_m=args.measurement_sigma_m,
        process_accel_sigma_mps2=args.process_accel_sigma_mps2,
        initial_position_sigma_m=args.initial_position_sigma_m,
        initial_velocity_sigma_mps=args.initial_velocity_sigma_mps,
        soft_residual_speed_mps=args.soft_residual_speed_mps,
        hard_residual_speed_mps=args.hard_residual_speed_mps,
        soft_noise_scale=args.soft_noise_scale,
        prefilter_window_size=args.prefilter_window_size,
        prefilter_lag_points=args.prefilter_lag_points,
        prefilter_soft_speed_mps=args.prefilter_soft_speed_mps,
        prefilter_hard_speed_mps=args.prefilter_hard_speed_mps,
        prefilter_hard_distance_m=args.prefilter_hard_distance_m,
        prefilter_hard_distance_dt_seconds=args.
        prefilter_hard_distance_dt_seconds,
        prefilter_bridge_neighbor_distance_m=args.
        prefilter_bridge_neighbor_distance_m,
        prefilter_bridge_center_distance_m=args.
        prefilter_bridge_center_distance_m,
        prefilter_burst_max_run_length=args.prefilter_burst_max_run_length,
        prefilter_median_window_size=args.prefilter_median_window_size,
    )
    plugin = DroneKalmanFilterPlugin(config=config)

    with args.input.open("r", encoding="utf-8") as source, args.output.open(
            "w", encoding="utf-8", newline="\n") as sink:
        for line in source:
            for output_line in plugin.process_json_line(line):
                sink.write(output_line)
                sink.write("\n")
        for output_line in plugin.flush_json():
            sink.write(output_line)
            sink.write("\n")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    """输出单个 JSONL 文件的基础统计报告。

    Args:
        args: 命令行解析后的参数对象。

    Returns:
        int: 进程退出码。
    """
    report = build_report(args.input)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _run_acceptance(args: argparse.Namespace) -> int:
    """生成 raw 与 smoothed 的验收指标摘要。

    Args:
        args: 命令行解析后的参数对象。

    Returns:
        int: 进程退出码。
    """
    # acceptance 只做离线验收，不会改任何业务输出文件。
    config = PluginConfig(
        lag_points=args.lag_points,
        min_dt_seconds=args.min_dt_seconds,
        max_segment_gap_seconds=args.max_segment_gap_seconds,
        idle_flush_seconds=args.idle_flush_seconds,
    )
    summary = write_acceptance_summary(
        raw_path=args.raw,
        smoothed_path=args.smoothed,
        output_path=args.output,
        config=config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """为 smooth 子命令补充可调的运行参数。

    Args:
        parser: 命令行参数解析器。

    Returns:
        None: 不返回值。
    """
    parser.add_argument("--smoother-mode",
                        choices=["kalman", "robust_prefilter_kalman"],
                        default=_DEFAULT_CONFIG.smoother_mode)
    parser.add_argument("--window-size",
                        type=int,
                        default=_DEFAULT_CONFIG.window_size)
    parser.add_argument("--lag-points",
                        type=int,
                        default=_DEFAULT_CONFIG.lag_points)
    parser.add_argument("--min-dt-seconds",
                        type=float,
                        default=_DEFAULT_CONFIG.min_dt_seconds)
    parser.add_argument("--max-segment-gap-seconds",
                        type=float,
                        default=_DEFAULT_CONFIG.max_segment_gap_seconds)
    parser.add_argument("--idle-flush-seconds",
                        type=float,
                        default=_DEFAULT_CONFIG.idle_flush_seconds)
    parser.add_argument("--measurement-sigma-m",
                        type=float,
                        default=_DEFAULT_CONFIG.measurement_sigma_m)
    parser.add_argument("--process-accel-sigma-mps2",
                        type=float,
                        default=_DEFAULT_CONFIG.process_accel_sigma_mps2)
    parser.add_argument("--initial-position-sigma-m",
                        type=float,
                        default=_DEFAULT_CONFIG.initial_position_sigma_m)
    parser.add_argument("--initial-velocity-sigma-mps",
                        type=float,
                        default=_DEFAULT_CONFIG.initial_velocity_sigma_mps)
    parser.add_argument("--soft-residual-speed-mps",
                        type=float,
                        default=_DEFAULT_CONFIG.soft_residual_speed_mps)
    parser.add_argument("--hard-residual-speed-mps",
                        type=float,
                        default=_DEFAULT_CONFIG.hard_residual_speed_mps)
    parser.add_argument("--soft-noise-scale",
                        type=float,
                        default=_DEFAULT_CONFIG.soft_noise_scale)
    parser.add_argument("--prefilter-window-size",
                        type=int,
                        default=_DEFAULT_CONFIG.prefilter_window_size)
    parser.add_argument("--prefilter-lag-points",
                        type=int,
                        default=_DEFAULT_CONFIG.prefilter_lag_points)
    parser.add_argument("--prefilter-soft-speed-mps",
                        type=float,
                        default=_DEFAULT_CONFIG.prefilter_soft_speed_mps)
    parser.add_argument("--prefilter-hard-speed-mps",
                        type=float,
                        default=_DEFAULT_CONFIG.prefilter_hard_speed_mps)
    parser.add_argument("--prefilter-hard-distance-m",
                        type=float,
                        default=_DEFAULT_CONFIG.prefilter_hard_distance_m)
    parser.add_argument(
        "--prefilter-hard-distance-dt-seconds",
        type=float,
        default=_DEFAULT_CONFIG.prefilter_hard_distance_dt_seconds,
    )
    parser.add_argument(
        "--prefilter-bridge-neighbor-distance-m",
        type=float,
        default=_DEFAULT_CONFIG.prefilter_bridge_neighbor_distance_m,
    )
    parser.add_argument(
        "--prefilter-bridge-center-distance-m",
        type=float,
        default=_DEFAULT_CONFIG.prefilter_bridge_center_distance_m,
    )
    parser.add_argument(
        "--prefilter-burst-max-run-length",
        type=int,
        default=_DEFAULT_CONFIG.prefilter_burst_max_run_length,
    )
    parser.add_argument(
        "--prefilter-median-window-size",
        type=int,
        default=_DEFAULT_CONFIG.prefilter_median_window_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
