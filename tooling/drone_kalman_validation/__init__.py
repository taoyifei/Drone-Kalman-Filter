from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "DiagnosticError",
    "PreflightError",
    "ValidationError",
    "generate_diagnostic_report",
    "generate_offline_rts_json_lines",
    "generate_offline_rts_messages",
    "generate_preflight_report",
    "generate_validation_report",
    "write_offline_rts_jsonl",
]


_EXPORT_MAP = {
    "DiagnosticError": ("drone_kalman_validation.diagnostics", "DiagnosticError"),
    "PreflightError": ("drone_kalman_validation.preflight", "PreflightError"),
    "ValidationError": ("drone_kalman_validation.validate", "ValidationError"),
    "generate_diagnostic_report": ("drone_kalman_validation.diagnostics", "generate_diagnostic_report"),
    "generate_offline_rts_json_lines": ("drone_kalman_validation.baseline", "generate_offline_rts_json_lines"),
    "generate_offline_rts_messages": ("drone_kalman_validation.baseline", "generate_offline_rts_messages"),
    "generate_preflight_report": ("drone_kalman_validation.preflight", "generate_preflight_report"),
    "generate_validation_report": ("drone_kalman_validation.validate", "generate_validation_report"),
    "write_offline_rts_jsonl": ("drone_kalman_validation.baseline", "write_offline_rts_jsonl"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
