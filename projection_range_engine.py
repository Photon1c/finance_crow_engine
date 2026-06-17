"""Projection window presets for Laser Falcon simulations."""

from __future__ import annotations

PROJECTION_PRESETS = (7, 14, 30, 60, 90, 180)
DEFAULT_PROJECTION_DAYS = 30
MIN_PROJECTION_DAYS = 1
MAX_PROJECTION_DAYS = 180


def clamp_projection_days(days: int) -> int:
    return max(MIN_PROJECTION_DAYS, min(int(days), MAX_PROJECTION_DAYS))
