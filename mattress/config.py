"""Configuration: packaged defaults <- optional user YAML <- --set overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")


class Cfg(dict):
    """dict with attribute access for nested sections (cfg.faces.blend)."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_cfg(d: dict) -> Cfg:
    return Cfg({k: _to_cfg(v) if isinstance(v, dict) else v for k, v in d.items()})


def apply_override(d: dict, expr: str) -> None:
    """Apply 'a.b.c=value' (value parsed as YAML scalar) in place."""
    if "=" not in expr:
        raise ValueError(f"--set expects key=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    parts = key.strip().split(".")
    node = d
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"unknown config section {p!r} in {key!r}")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"unknown config key {key!r}")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(user_path: str | Path | None = None, overrides: list[str] | None = None) -> Cfg:
    with open(DEFAULTS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if user_path:
        with open(user_path, encoding="utf-8") as f:
            data = _merge(data, yaml.safe_load(f) or {})
    for expr in overrides or []:
        apply_override(data, expr)
    return _to_cfg(data)
