"""Best-effort PyPI version check (stdlib only, no extra dependency)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

PYPI_URL = "https://pypi.org/pypi/kgrp/json"
_USER_AGENT = "kgrp-update-check"


def _parse_version(v: str) -> tuple[int, ...]:
    """Loose numeric parse good enough for '0.2.4' style versions."""
    parts: list[int] = []
    for chunk in str(v).split("."):
        num = ""
        for ch in chunk:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts)


def get_latest_version(timeout: float = 5.0) -> str:
    """Return the latest version string published on PyPI for kgrp."""
    req = urllib.request.Request(
        PYPI_URL, headers={"User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["info"]["version"])


def is_update_available(current: str, latest: str | None) -> bool:
    if not latest:
        return False
    try:
        return _parse_version(latest) > _parse_version(current)
    except Exception:
        return False
