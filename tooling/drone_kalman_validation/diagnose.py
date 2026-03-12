from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from drone_kalman_filter.cli import _DEFAULT_CONFIG
from drone_kalman_filter.config import PluginConfig

from .diagnostics import generate_diagnostic_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose current smoothing against a fixed-interval RTS baseline.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--smoothed", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-output")
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--min-segment-points", type=int, default=5)
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
    )
    summary = generate_diagnostic_report(
        raw_path=args.raw,
        smoothed_path=args.smoothed,
        out_dir=args.out_dir,
        config=config,
        baseline_output_path=args.baseline_output,
        max_segments=args.max_segments,
        min_segment_points=args.min_segment_points,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
