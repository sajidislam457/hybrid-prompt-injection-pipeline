"""
Importance weights for Lab / team-labeled training rows (Layer 2).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

DEFAULT_TEAM_SOURCES = frozenset({"review_queue", "team_train", "inbox_review", "inbox_manual"})


def team_sample_weight(source: str, *, team_weight: float, default_weight: float = 1.0) -> float:
    src = (source or "").strip().lower()
    if src in DEFAULT_TEAM_SOURCES:
        return float(team_weight)
    return float(default_weight)


def build_sample_weights(
    samples: Sequence[Dict[str, Any]],
    *,
    team_weight: float = 50.0,
    default_weight: float = 1.0,
    team_sources: Sequence[str] | None = None,
) -> Tuple[List[float], Dict[str, int]]:
    sources = frozenset(s.lower() for s in (team_sources or DEFAULT_TEAM_SOURCES))
    weights: List[float] = []
    stats = {"team_rows": 0, "bulk_rows": 0}
    for row in samples:
        src = (row.get("source") or "").strip().lower()
        if src in sources:
            weights.append(float(team_weight))
            stats["team_rows"] += 1
        else:
            weights.append(float(default_weight))
            stats["bulk_rows"] += 1
    return weights, stats
