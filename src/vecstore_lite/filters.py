from __future__ import annotations

from typing import Any


def match_metadata(meta: dict[str, Any], filt: dict[str, Any] | None) -> bool:
    """支持等值、`$in`、`$gte` / `$lte` 的元数据过滤。

    示例::

        {"tag": "demo"}
        {"tag": {"$in": ["demo", "prod"]}}
        {"score": {"$gte": 0.5, "$lte": 1.0}}
        {"lang": "zh", "year": {"$gte": 2024}}
    """
    if not filt:
        return True
    for key, expected in filt.items():
        actual = meta.get(key)
        if isinstance(expected, dict) and any(str(k).startswith("$") for k in expected):
            if "$in" in expected:
                if actual not in expected["$in"]:
                    return False
            if "$gte" in expected:
                if actual is None or actual < expected["$gte"]:
                    return False
            if "$lte" in expected:
                if actual is None or actual > expected["$lte"]:
                    return False
            if "$ne" in expected:
                if actual == expected["$ne"]:
                    return False
        else:
            if actual != expected:
                return False
    return True
