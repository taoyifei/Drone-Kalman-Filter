from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "CompareError",
    "generate_compare_report",
]


_EXPORT_MAP = {
    "CompareError": ("drone_kalman_visualization.compare", "CompareError"),
    "generate_compare_report": ("drone_kalman_visualization.compare", "generate_compare_report"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
