from __future__ import annotations

import yaml


def parse_overrides(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        key, value = item.split("=", 1)
        model, param = key.split(".", 1)
        out.setdefault(model, {})[param] = yaml.safe_load(value)
    return out
