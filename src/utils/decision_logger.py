"""
Phase 5: Structured decision logging for upgrade observability.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DecisionLogger:
    def __init__(self, log_path: str = "logs/decisions.jsonl", enabled: bool = True):
        self.enabled = enabled
        self.log_path = Path(log_path)
        self._fh = None
        if enabled:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Decision logger path issue: %s", exc)
                self.enabled = False

    def log(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        try:
            # Keep one append handle open during long runs to avoid open/close thrash.
            if self._fh is None:
                self._fh = self.log_path.open("a", encoding="utf-8")
            self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            # Flush occasionally so logs survive crashes without per-call fsync cost.
            if getattr(self, "_n", 0) % 50 == 0:
                self._fh.flush()
            self._n = getattr(self, "_n", 0) + 1
        except Exception as exc:
            logger.warning("Failed to write decision log: %s", exc)
            try:
                if self._fh:
                    self._fh.close()
            except Exception:
                pass
            self._fh = None

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None
