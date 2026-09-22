"""YAML configuration with `_base_` inheritance and dotted command-line overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import yaml

BASE_KEY = "_base_"
PathLike = Union[str, os.PathLike]


class Config(dict):
    """Nested dict with attribute access. Missing attributes raise AttributeError."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def __delattr__(self, name: str) -> None:
        del self[name]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        return cls({k: _wrap(v) for k, v in data.items()})

    def to_dict(self) -> dict:
        return _unwrap(self)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Mapping):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = _wrap(value)

    def copy(self) -> "Config":  # type: ignore[override]
        return Config.from_dict(copy.deepcopy(self.to_dict()))


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursive merge; scalars and lists in `override` replace those in `base`."""
    out = dict(copy.deepcopy(_unwrap(base)))
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(_unwrap(value))
    return out


def load_yaml(path: PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_config(path: PathLike, overrides: Optional[Iterable[str]] = None) -> Config:
    """Load a YAML file, resolve `_base_` files (relative to the file) and apply overrides.

    Overrides use the form ``section.key=value``; the value is parsed as YAML,
    so ``training.lr=1e-4`` gives a float and ``data.encoder=uni`` a string.
    """
    path = Path(path)
    raw = load_yaml(path)
    bases = raw.pop(BASE_KEY, None)
    merged: dict = {}
    if bases:
        if isinstance(bases, str):
            bases = [bases]
        for base in bases:
            base_path = Path(base) if os.path.isabs(base) else (path.parent / base)
            merged = merge(merged, load_config(base_path).to_dict())
    merged = merge(merged, raw)
    cfg = Config.from_dict(merged)
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        cfg.set_path(key.strip(), yaml.safe_load(value))
    return cfg


def save_config(cfg: Mapping[str, Any], path: PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(_unwrap(cfg), fh, sort_keys=False, allow_unicode=True)


def resolve_encoder_config(cfg: Config, configs_dir: PathLike = "configs") -> Config:
    """Return the encoder YAML referenced by ``cfg.data.encoder``.

    ``resnet50_raw`` and similar suffixed names reuse the base encoder file;
    the suffix only distinguishes the feature cache directory.
    """
    name = str(cfg.data.encoder)
    base = name.split("_raw")[0]
    enc_path = Path(configs_dir) / "encoders" / f"{base}.yaml"
    if not enc_path.exists():
        raise FileNotFoundError(f"no encoder config for {name!r} at {enc_path}")
    enc = load_config(enc_path)
    enc["cache_name"] = name
    return enc


def as_list(value: Union[Sequence[Any], Any]) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
