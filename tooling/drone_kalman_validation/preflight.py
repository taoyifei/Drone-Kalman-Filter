from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from drone_kalman_filter.message import parse_time


_PREFERRED_FAMILY_SPECS = [
    (
        "1581F6Q8D246J00GJJ55",
        ["fusion", "5_6_HG0001", "734057F6925A41F5"],
        "同一目标、三种设备节奏，最适合做首批前端观感验证",
    ),
    (
        "1581F5FHC257L00DKK9U",
        ["fusion", "5_6_HG0001", "734057F6925A41F5"],
        "长轨、跨设备、gap 更复杂，适合测平滑不失真",
    ),
    (
        "1581F6N8C23BE0036ELZ",
        ["fusion", "5_6_HG0001", "734057F6925A41F5"],
        "高速度或高抖动代表样本，优先验证抗跳变",
    ),
    (
        "1581F6GKX256K0040BG4",
        ["fusion", "734057F6925A41F5"],
        "极端碎片化样本，优先验证长 gap 不乱连",
    ),
]
_PREFERRED_SHORT_LIVED_TARGETS = [
    ("1_3_00001_3240", "1_3_00001"),
    ("1_3_00001_3828", "1_3_00001"),
]
_DEFAULT_MAX_SEGMENT_GAP_SECONDS = 10.0


@dataclass(slots=True)
class _TrackAccumulator:
    """累积单条 `(targetId, deviceId)` 轨迹的统计量。"""

    target_id: str | None
    family_id: str | None
    device_id: str | None
    message_count: int = 0
    segment_count: int = 0
    valid_position_count: int = 0
    gap_count: int = 0
    max_gap_seconds: float = 0.0
    first_event_time: datetime | None = None
    last_event_time: datetime | None = None
    last_valid_event_time: datetime | None = None
    last_was_valid: bool = False
    dt_values: list[float] = field(default_factory=list)
    speed_values: list[float] = field(default_factory=list)


@dataclass(slots=True)
class _DeviceAccumulator:
    """累积单个设备维度的节奏和速度字段统计。"""

    device_id: str | None
    message_count: int = 0
    pair_ids: set[tuple[str | None, str | None]] = field(default_factory=set)
    extension_signatures: Counter[str] = field(default_factory=Counter)
    last_event_time: datetime | None = None
    dt_values: list[float] = field(default_factory=list)
    max_gap_seconds: float = 0.0
    speed_non_negative_count: int = 0
    speed_negative_count: int = 0
    speed_invalid_count: int = 0


class PreflightError(ValueError):
    """表示 preflight 输入或报告生成失败。"""


def generate_preflight_report(
    input_path: str | Path,
    out_dir: str | Path,
    *,
    max_segment_gap_seconds: float = _DEFAULT_MAX_SEGMENT_GAP_SECONDS,
) -> dict[str, Any]:
    """扫描 raw JSONL 并生成落地前分析报告。"""

    summary, candidate_tracks = _scan_input(
        input_path,
        max_segment_gap_seconds=max_segment_gap_seconds,
    )
    report_markdown = _build_report_markdown(summary, candidate_tracks)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "candidate_tracks.json").write_text(
        json.dumps(candidate_tracks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(report_markdown, encoding="utf-8")
    return summary


def main() -> int:
    """作为 CLI 入口生成 preflight 报告。"""

    parser = argparse.ArgumentParser(
        description="Analyze raw JSONL before running smoothing validation.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--max-segment-gap-seconds",
        type=float,
        default=_DEFAULT_MAX_SEGMENT_GAP_SECONDS,
    )
    args = parser.parse_args()
    summary = generate_preflight_report(
        input_path=args.input,
        out_dir=args.out_dir,
        max_segment_gap_seconds=args.max_segment_gap_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _scan_input(
    input_path: str | Path,
    *,
    max_segment_gap_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """按原始行顺序扫描输入并汇总统计。"""

    path = Path(input_path)
    if not path.exists():
        raise PreflightError(f"Input file does not exist: {path}")

    device_accumulators: dict[str | None, _DeviceAccumulator] = {}
    track_accumulators: dict[tuple[str | None, str | None], _TrackAccumulator] = {}
    family_targets: dict[str | None, set[str | None]] = defaultdict(set)
    family_devices: dict[str | None, Counter[str | None]] = defaultdict(Counter)

    message_count = 0
    trace_id_null_count = 0
    global_start: datetime | None = None
    global_end: datetime | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = _parse_json_line(line, line_no)
            identity = payload.get("identity") or {}
            source = payload.get("source") or {}
            spatial = payload.get("spatial") or {}
            position = spatial.get("position") or {}
            velocity = spatial.get("velocity") or {}

            target_id = identity.get("targetId")
            device_id = source.get("deviceId")
            trace_id = identity.get("traceId")
            family_id = _normalize_family_id(target_id)
            event_time = _require_event_time(payload.get("eventTime"), line_no)
            latitude = _as_float(position.get("latitude"))
            longitude = _as_float(position.get("longitude"))
            speed_total = _as_float(velocity.get("speedTotal"))
            extension_signature = _extension_signature(payload.get("extension"))

            message_count += 1
            if trace_id in (None, ""):
                trace_id_null_count += 1
            global_start = event_time if global_start is None else min(global_start, event_time)
            global_end = event_time if global_end is None else max(global_end, event_time)

            _update_device_accumulator(
                device_accumulators,
                device_id=device_id,
                target_id=target_id,
                event_time=event_time,
                speed_total=speed_total,
                extension_signature=extension_signature,
            )
            _update_track_accumulator(
                track_accumulators,
                target_id=target_id,
                family_id=family_id,
                device_id=device_id,
                event_time=event_time,
                latitude=latitude,
                longitude=longitude,
                speed_total=speed_total,
                max_segment_gap_seconds=max_segment_gap_seconds,
            )
            family_targets[family_id].add(target_id)
            family_devices[family_id][device_id] += 1

    if message_count == 0:
        raise PreflightError("Input file contains no JSON messages.")

    track_stats = [
        _finalize_track_stats(accumulator)
        for accumulator in track_accumulators.values()
    ]
    device_stats = [
        _finalize_device_stats(accumulator)
        for accumulator in device_accumulators.values()
    ]
    device_stats.sort(key=lambda item: (-item["message_count"], item["deviceId"] or ""))

    family_stats = _build_family_stats(family_targets, family_devices)
    candidate_tracks = _build_candidate_tracks(track_stats, family_stats)
    first_wave_candidates = _build_first_wave_candidates(
        family_stats,
        track_stats,
        candidate_tracks,
    )

    summary = {
        "message_count": message_count,
        "target_count": len({stat["targetId"] for stat in track_stats}),
        "device_count": len(device_stats),
        "time_range": {
            "start": _format_datetime(global_start),
            "end": _format_datetime(global_end),
        },
        "trace_id_null_rate": round(trace_id_null_count / message_count, 6),
        "by_device": device_stats,
        "stream_risks": _build_stream_risks(
            device_stats=device_stats,
            family_stats=family_stats,
            track_stats=track_stats,
            trace_id_null_rate=trace_id_null_count / message_count,
        ),
        "first_wave_candidates": first_wave_candidates,
    }
    return summary, candidate_tracks


def _update_device_accumulator(
    device_accumulators: dict[str | None, _DeviceAccumulator],
    *,
    device_id: str | None,
    target_id: str | None,
    event_time: datetime,
    speed_total: float | None,
    extension_signature: str,
) -> None:
    """更新设备维度的节奏与字段质量统计。"""

    accumulator = device_accumulators.setdefault(
        device_id,
        _DeviceAccumulator(device_id=device_id),
    )
    accumulator.message_count += 1
    accumulator.pair_ids.add((target_id, device_id))
    accumulator.extension_signatures[extension_signature] += 1
    if accumulator.last_event_time is not None:
        delta = (event_time - accumulator.last_event_time).total_seconds()
        if delta > 0.0:
            accumulator.dt_values.append(delta)
            accumulator.max_gap_seconds = max(accumulator.max_gap_seconds, delta)
    accumulator.last_event_time = event_time

    if speed_total is None:
        accumulator.speed_invalid_count += 1
    elif speed_total >= 0.0:
        accumulator.speed_non_negative_count += 1
    else:
        accumulator.speed_negative_count += 1


def _update_track_accumulator(
    track_accumulators: dict[tuple[str | None, str | None], _TrackAccumulator],
    *,
    target_id: str | None,
    family_id: str | None,
    device_id: str | None,
    event_time: datetime,
    latitude: float | None,
    longitude: float | None,
    speed_total: float | None,
    max_segment_gap_seconds: float,
) -> None:
    """更新单条轨迹的长度、切段和节奏统计。"""

    key = (target_id, device_id)
    accumulator = track_accumulators.setdefault(
        key,
        _TrackAccumulator(
            target_id=target_id,
            family_id=family_id,
            device_id=device_id,
        ),
    )
    accumulator.message_count += 1
    accumulator.first_event_time = (
        event_time
        if accumulator.first_event_time is None
        else min(accumulator.first_event_time, event_time)
    )
    accumulator.last_event_time = (
        event_time
        if accumulator.last_event_time is None
        else max(accumulator.last_event_time, event_time)
    )
    if speed_total is not None and speed_total >= 0.0:
        accumulator.speed_values.append(speed_total)

    has_valid_position = latitude is not None and longitude is not None
    if not has_valid_position:
        accumulator.last_was_valid = False
        return

    accumulator.valid_position_count += 1
    if not accumulator.last_was_valid:
        accumulator.segment_count += 1
    elif accumulator.last_valid_event_time is not None:
        delta = (event_time - accumulator.last_valid_event_time).total_seconds()
        if delta <= 0.0 or delta > max_segment_gap_seconds:
            accumulator.segment_count += 1
            if delta > max_segment_gap_seconds:
                accumulator.gap_count += 1
                accumulator.max_gap_seconds = max(accumulator.max_gap_seconds, delta)
        else:
            accumulator.dt_values.append(delta)
    accumulator.last_valid_event_time = event_time
    accumulator.last_was_valid = True


def _finalize_device_stats(accumulator: _DeviceAccumulator) -> dict[str, Any]:
    """把设备累积统计整理成 JSON 结果。"""

    total_speed = (
        accumulator.speed_non_negative_count
        + accumulator.speed_negative_count
        + accumulator.speed_invalid_count
    )
    top_signatures = [
        {"signature": name, "count": count}
        for name, count in accumulator.extension_signatures.most_common(3)
    ]
    return {
        "deviceId": accumulator.device_id,
        "message_count": accumulator.message_count,
        "pair_count": len(accumulator.pair_ids),
        "dt_median_s": _round_or_none(_safe_median(accumulator.dt_values)),
        "dt_p95_s": _round_or_none(_percentile(accumulator.dt_values, 0.95)),
        "max_gap_s": _round_or_none(accumulator.max_gap_seconds),
        "speed_parseability": {
            "non_negative_count": accumulator.speed_non_negative_count,
            "negative_count": accumulator.speed_negative_count,
            "invalid_count": accumulator.speed_invalid_count,
            "non_negative_rate": _round_or_none(
                _safe_ratio(accumulator.speed_non_negative_count, total_speed),
            ),
            "negative_rate": _round_or_none(
                _safe_ratio(accumulator.speed_negative_count, total_speed),
            ),
            "invalid_rate": _round_or_none(
                _safe_ratio(accumulator.speed_invalid_count, total_speed),
            ),
        },
        "extension_signatures": top_signatures,
    }


def _finalize_track_stats(accumulator: _TrackAccumulator) -> dict[str, Any]:
    """把轨迹累积统计整理成 JSON 结果。"""

    return {
        "targetId": accumulator.target_id,
        "familyId": accumulator.family_id,
        "deviceId": accumulator.device_id,
        "message_count": accumulator.message_count,
        "valid_position_count": accumulator.valid_position_count,
        "segment_count": accumulator.segment_count,
        "gap_count": accumulator.gap_count,
        "dt_median_s": _round_or_none(_safe_median(accumulator.dt_values)),
        "dt_p95_s": _round_or_none(_percentile(accumulator.dt_values, 0.95)),
        "max_gap_s": _round_or_none(accumulator.max_gap_seconds),
        "speed_p95_mps": _round_or_none(_percentile(accumulator.speed_values, 0.95)),
        "speed_p99_mps": _round_or_none(_percentile(accumulator.speed_values, 0.99)),
        "start_time": _format_datetime(accumulator.first_event_time),
        "end_time": _format_datetime(accumulator.last_event_time),
    }


def _build_family_stats(
    family_targets: dict[str | None, set[str | None]],
    family_devices: dict[str | None, Counter[str | None]],
) -> list[dict[str, Any]]:
    """按去前缀后的 family 维度整理多设备家族。"""

    families = []
    for family_id, targets in family_targets.items():
        counts = family_devices[family_id]
        families.append(
            {
                "familyId": family_id,
                "device_count": len(counts),
                "devices": sorted(device for device in counts if device is not None),
                "targetIds": sorted(target for target in targets if target is not None),
                "total_message_count": sum(counts.values()),
                "by_device": dict(
                    sorted(
                        counts.items(),
                        key=lambda item: (-item[1], item[0] or ""),
                    ),
                ),
            }
        )
    families.sort(
        key=lambda item: (
            -item["device_count"],
            -item["total_message_count"],
            item["familyId"] or "",
        ),
    )
    return families


def _build_candidate_tracks(
    track_stats: list[dict[str, Any]],
    family_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    """按问题类型整理候选轨迹和目标家族。"""

    dense_tracks = sorted(
        track_stats,
        key=lambda item: (
            -item["message_count"],
            item["deviceId"] or "",
            item["targetId"] or "",
        ),
    )[:12]
    fragmented_tracks = sorted(
        [item for item in track_stats if item["message_count"] >= 100],
        key=lambda item: (
            -item["segment_count"],
            -(item["dt_p95_s"] or 0.0),
            -(item["max_gap_s"] or 0.0),
        ),
    )[:12]
    jitter_tracks = sorted(
        [item for item in track_stats if item["message_count"] >= 100],
        key=lambda item: (
            -(item["speed_p95_mps"] or -1.0),
            -(item["speed_p99_mps"] or -1.0),
            -item["message_count"],
        ),
    )[:12]
    clutter_tracks = sorted(
        [
            item
            for item in track_stats
            if item["deviceId"] == "1_3_00001" and 20 <= item["message_count"] <= 80
        ],
        key=lambda item: (
            -(item["speed_p95_mps"] or -1.0),
            -item["message_count"],
            item["targetId"] or "",
        ),
    )[:12]
    cross_device_families = [item for item in family_stats if item["device_count"] >= 2][:12]
    return {
        "high_density_long_tracks": dense_tracks,
        "gap_heavy_fragmented_tracks": fragmented_tracks,
        "high_speed_or_high_jitter_tracks": jitter_tracks,
        "short_lived_clutter_tracks": clutter_tracks,
        "cross_device_same_family_tracks": cross_device_families,
    }


def _build_first_wave_candidates(
    family_stats: list[dict[str, Any]],
    track_stats: list[dict[str, Any]],
    candidate_tracks: dict[str, Any],
) -> list[dict[str, Any]]:
    """组装首批验证样本清单。"""

    family_map = {item["familyId"]: item for item in family_stats}
    track_map = {
        (item["targetId"], item["deviceId"]): item
        for item in track_stats
    }
    selected: list[dict[str, Any]] = []
    selected_families: set[str | None] = set()

    for family_id, preferred_devices, use_case in _PREFERRED_FAMILY_SPECS:
        family = family_map.get(family_id)
        if family is None:
            continue
        selected.append(
            _build_family_candidate(
                family,
                track_map=track_map,
                preferred_devices=preferred_devices,
                use_case=use_case,
            ),
        )
        selected_families.add(family_id)

    if not selected:
        for family in candidate_tracks["cross_device_same_family_tracks"][:4]:
            selected.append(
                _build_family_candidate(
                    family,
                    track_map=track_map,
                    preferred_devices=family["devices"],
                    use_case="多设备同源家族，适合做第一批前端观感验证",
                ),
            )
            selected_families.add(family["familyId"])

    fallback_clutter = iter(candidate_tracks["short_lived_clutter_tracks"])
    for target_id, device_id in _PREFERRED_SHORT_LIVED_TARGETS:
        track = track_map.get((target_id, device_id))
        if track is None:
            track = next(fallback_clutter, None)
        if track is None:
            continue
        selected.append(
            {
                "kind": "track",
                "targetId": track["targetId"],
                "deviceId": track["deviceId"],
                "message_count": track["message_count"],
                "segment_count": track["segment_count"],
                "dt_median_s": track["dt_median_s"],
                "dt_p95_s": track["dt_p95_s"],
                "speed_p95_mps": track["speed_p95_mps"],
                "use_case": "短命高杂波目标，验证不会乱改和不会污染状态",
            }
        )
    return selected


def _build_family_candidate(
    family: dict[str, Any],
    *,
    track_map: dict[tuple[str | None, str | None], dict[str, Any]],
    preferred_devices: list[str],
    use_case: str,
) -> dict[str, Any]:
    """把 family 统计转换成首批验证候选。"""

    per_device_tracks = []
    for device_id in preferred_devices:
        preferred_target = f"f_{family['familyId']}" if device_id == "fusion" else family["familyId"]
        track = track_map.get((preferred_target, device_id))
        if track is None:
            for candidate_target in family["targetIds"]:
                track = track_map.get((candidate_target, device_id))
                if track is not None:
                    break
        if track is None:
            continue
        per_device_tracks.append(
            {
                "targetId": track["targetId"],
                "deviceId": track["deviceId"],
                "message_count": track["message_count"],
                "segment_count": track["segment_count"],
                "dt_median_s": track["dt_median_s"],
                "dt_p95_s": track["dt_p95_s"],
                "speed_p95_mps": track["speed_p95_mps"],
            }
        )
    return {
        "kind": "family",
        "familyId": family["familyId"],
        "targetIds": family["targetIds"],
        "device_count": family["device_count"],
        "devices": per_device_tracks,
        "use_case": use_case,
    }


def _build_stream_risks(
    *,
    device_stats: list[dict[str, Any]],
    family_stats: list[dict[str, Any]],
    track_stats: list[dict[str, Any]],
    trace_id_null_rate: float,
) -> list[dict[str, Any]]:
    """把当前数据触发的实时风险整理成结论。"""

    risks = []
    if trace_id_null_rate >= 0.99:
        risks.append(
            {
                "code": "trace_id_unavailable",
                "severity": "high",
                "summary": "traceId 基本不可用，切段只能依赖 eventTime gap。",
                "evidence": {"trace_id_null_rate": round(trace_id_null_rate, 6)},
            }
        )

    dense_devices = [
        item["deviceId"]
        for item in device_stats
        if item["dt_median_s"] is not None and item["dt_median_s"] <= 0.6
    ]
    sparse_devices = [
        item["deviceId"]
        for item in device_stats
        if item["dt_p95_s"] is not None and item["dt_p95_s"] > 10.0
    ]
    if dense_devices and sparse_devices:
        risks.append(
            {
                "code": "mixed_device_cadence",
                "severity": "high",
                "summary": "高密度设备与长 gap 设备并存，固定滞后体验会明显分化。",
                "evidence": {
                    "dense_devices": dense_devices,
                    "sparse_devices": sparse_devices,
                },
            }
        )

    family_overlap = [item for item in family_stats if item["device_count"] >= 2]
    if family_overlap:
        risks.append(
            {
                "code": "cross_device_same_family",
                "severity": "medium",
                "summary": "存在同源 family 跨设备出现，验证时必须按 device 分开看。",
                "evidence": {
                    "family_count": len(family_overlap),
                    "top_family": family_overlap[0]["familyId"],
                },
            }
        )

    speed_issues = [
        item
        for item in device_stats
        if item["speed_parseability"]["negative_rate"] is not None
        and (
            item["speed_parseability"]["negative_rate"] > 0.1
            or item["speed_parseability"]["invalid_rate"] > 0.1
        )
    ]
    if speed_issues:
        risks.append(
            {
                "code": "speed_fields_unreliable",
                "severity": "medium",
                "summary": "speedTotal 可解析性和符号稳定性不足，后续验证应继续以位置为主观测。",
                "evidence": {
                    "affected_devices": [item["deviceId"] for item in speed_issues],
                },
            }
        )

    short_lived_tracks = [
        item
        for item in track_stats
        if item["deviceId"] == "1_3_00001" and item["message_count"] <= 10
    ]
    if short_lived_tracks:
        risks.append(
            {
                "code": "short_lived_clutter_tracks",
                "severity": "medium",
                "summary": "存在大量短命目标，需验证不会串轨、不会乱改和不会污染状态。",
                "evidence": {
                    "deviceId": "1_3_00001",
                    "short_track_count": len(short_lived_tracks),
                },
            }
        )
    return risks


def _build_report_markdown(
    summary: dict[str, Any],
    candidate_tracks: dict[str, Any],
) -> str:
    """把 preflight 结果渲染成面向实施的 Markdown 报告。"""

    lines = [
        "# raw2 验证前数据分析报告",
        "",
        "## 1. 全局概况",
        "",
        f"- 消息数：`{summary['message_count']}`",
        f"- 设备数：`{summary['device_count']}`",
        f"- 目标数：`{summary['target_count']}`",
        f"- 时间范围：`{summary['time_range']['start']}` 到 `{summary['time_range']['end']}`",
        f"- traceId 空值率：`{summary['trace_id_null_rate']}`",
        "",
        "## 2. 设备节奏画像",
        "",
        "| deviceId | message_count | pair_count | dt_median_s | dt_p95_s | max_gap_s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for device in summary["by_device"]:
        lines.append(
            "| "
            f"{device['deviceId']} | "
            f"{device['message_count']} | "
            f"{device['pair_count']} | "
            f"{_display(device['dt_median_s'])} | "
            f"{_display(device['dt_p95_s'])} | "
            f"{_display(device['max_gap_s'])} |"
        )

    lines.extend(
        [
            "",
            "## 3. 当前流会触发的风险",
            "",
        ]
    )
    for risk in summary["stream_risks"]:
        lines.append(
            f"- `{risk['code']}`（{risk['severity']}）：{risk['summary']} "
            f"证据：`{json.dumps(risk['evidence'], ensure_ascii=False)}`"
        )

    lines.extend(
        [
            "",
            "## 4. f_... 与非 f_... 同源家族",
            "",
            "这些 family 当前只做同源标记，不做业务融合；后续验证仍按 "
            "`(targetId, deviceId)` 分开回放。",
            "",
        ]
    )
    for family in candidate_tracks["cross_device_same_family_tracks"][:8]:
        lines.append(
            f"- `{family['familyId']}`：devices={family['devices']}，"
            f"targetIds={family['targetIds']}，"
            f"total_message_count={family['total_message_count']}"
        )

    lines.extend(
        [
            "",
            "## 5. 第一批建议验证样本",
            "",
        ]
    )
    for candidate in summary["first_wave_candidates"]:
        if candidate["kind"] == "family":
            lines.append(
                f"- family `{candidate['familyId']}`：{candidate['use_case']}"
            )
            for device in candidate["devices"]:
                lines.append(
                    "  - "
                    f"{device['deviceId']} / {device['targetId']}："
                    f"{device['message_count']} 点，"
                    f"{device['segment_count']} 段，"
                    f"dt_median={_display(device['dt_median_s'])}s"
                )
        else:
            lines.append(
                f"- track `{candidate['targetId']}` / `{candidate['deviceId']}`："
                f"{candidate['use_case']}，"
                f"{candidate['message_count']} 点，"
                f"{candidate['segment_count']} 段"
            )

    lines.extend(
        [
            "",
            "## 6. 下一轮验证执行约束",
            "",
            "- 必须保留 `raw2` 原始行顺序逐条送入插件，模拟真实实时交错流。",
            "- 不允许按 `targetId` 先排序再喂给算法。",
            "- 只允许修改 `spatial.position.latitude/longitude`。",
            "- 其他字段必须透传，包括 `identity/source/lifecycle/extension/"
            "eventTime/processTime/originalData`。",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_json_line(line: str, line_no: int) -> dict[str, Any]:
    """解析单行 JSON 并在失败时抛出可读异常。"""

    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise PreflightError(
            f"Invalid JSON at line {line_no}.",
        ) from exc


def _require_event_time(value: Any, line_no: int) -> datetime:
    """解析 eventTime 并保证其可用。"""

    event_time = parse_time(value)
    if event_time is None:
        raise PreflightError(f"Invalid eventTime at line {line_no}.")
    return event_time


def _normalize_family_id(target_id: str | None) -> str | None:
    """把 fusion 的 `f_...` 目标映射到同源 family。"""

    if isinstance(target_id, str) and target_id.startswith("f_"):
        return target_id[2:]
    return target_id


def _extension_signature(extension: Any) -> str:
    """提取 extension 的顶层结构签名。"""

    if extension is None:
        return "null"
    if isinstance(extension, dict):
        return ",".join(sorted(extension.keys())) or "empty-dict"
    return type(extension).__name__


def _safe_median(values: Iterable[float]) -> float | None:
    """在列表为空时返回 None，而不是抛异常。"""

    values = list(values)
    if not values:
        return None
    return float(median(values))


def _percentile(values: list[float], probability: float) -> float | None:
    """计算简单分位数。"""

    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    """在分母为零时返回 None。"""

    if denominator == 0:
        return None
    return numerator / denominator


def _round_or_none(value: float | None) -> float | None:
    """统一浮点输出精度。"""

    if value is None:
        return None
    return round(value, 6)


def _format_datetime(value: datetime | None) -> str | None:
    """把 UTC 时间转成稳定的 ISO-8601 文本。"""

    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


def _as_float(value: Any) -> float | None:
    """把字符串或数字安全转换成浮点数。"""

    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _display(value: Any) -> str:
    """把空值安全展示成 markdown 文本。"""

    return "-" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
