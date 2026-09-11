"""
Exact prompt overrides applied only after Train.

Team-reviewed examples live here so chat users get BLOCK + the type the team set.
Layer 4 / pattern scoring must not override these.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.malicious_inbox import fingerprint

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = ROOT / "data" / "team_overrides.json"
_LOCK = threading.Lock()
_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_raw() -> Dict[str, Any]:
    if not STORE_PATH.exists():
        return {"examples": {}}
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data.get("examples"), dict):
            data["examples"] = {}
        return data
    except Exception as exc:
        logger.warning("team overrides load failed: %s", exc)
        return {"examples": {}}


def _save(data: Dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STORE_PATH)


def load_map() -> Dict[str, Dict[str, Any]]:
    global _cache
    with _LOCK:
        data = _load_raw()
        if not data.get("examples"):
            seeded = _seed_from_attack_bank(data)
            if seeded:
                _save(data)
        _cache = dict(data.get("examples") or {})
        return dict(_cache)


def _seed_from_attack_bank(data: Dict[str, Any]) -> int:
    bank_path = ROOT / "data" / "attack_bank.json"
    if not bank_path.exists():
        return 0
    try:
        raw = json.loads(bank_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(raw, list):
        return 0
    n = 0
    examples = data.setdefault("examples", {})
    ts = _now()
    for row in raw:
        if (row.get("source") or "") != "team_train":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        fp = fingerprint(text)
        examples[fp] = {
            "fingerprint": fp,
            "prompt": text,
            "attack_type": row.get("attack_type") or "unknown",
            "updated_at": ts,
            "source": "team_train",
        }
        n += 1
    return n


def invalidate() -> None:
    global _cache
    with _LOCK:
        _cache = None


def match(text: str) -> Optional[Dict[str, Any]]:
    global _cache
    prompt = (text or "").strip()
    if not prompt:
        return None
    fp = fingerprint(prompt)
    with _LOCK:
        examples = _cache
        if examples is None:
            examples = dict(_load_raw().get("examples") or {})
            _cache = examples
        hit = examples.get(fp)
        return dict(hit) if hit else None


def apply_review_items(items: List[Dict[str, Any]]) -> int:
    """Upsert label=1, remove label=0. Called only from Train."""
    global _cache
    n = 0
    ts = _now()
    with _LOCK:
        data = _load_raw()
        examples = data["examples"]
        for it in items:
            text = (it.get("prompt") or it.get("text") or "").strip()
            if not text:
                continue
            fp = fingerprint(text)
            label = int(it.get("label", 1))
            if label != 1:
                if fp in examples:
                    del examples[fp]
                    n += 1
                continue
            examples[fp] = {
                "fingerprint": fp,
                "prompt": text,
                "attack_type": (it.get("attack_type") or "unknown").strip() or "unknown",
                "updated_at": ts,
                "source": "team_train",
            }
            n += 1
        if n:
            data["examples"] = examples
            _save(data)
        _cache = dict(examples)
    return n


def lookup_trained(text: str) -> Dict[str, Any]:
    """
    Check whether a prompt was already team-trained / banked / in train.jsonl.
    Priority: team_overrides → attack_bank → train.jsonl.
    """
    prompt = (text or "").strip()
    empty = {
        "trained": False,
        "fingerprint": "",
        "prompt": prompt,
        "attack_type": None,
        "attack_display_name": None,
        "label": None,
        "source": None,
        "updated_at": None,
    }
    if not prompt:
        return empty
    fp = fingerprint(prompt)
    empty["fingerprint"] = fp

    hit = match(prompt)
    if hit:
        atype = (hit.get("attack_type") or "unknown").strip() or "unknown"
        return {
            "trained": True,
            "fingerprint": fp,
            "prompt": hit.get("prompt") or prompt,
            "attack_type": atype,
            "attack_display_name": _display_name(atype),
            "label": 1,
            "source": hit.get("source") or "team_overrides",
            "updated_at": hit.get("updated_at"),
        }

    bank_path = ROOT / "data" / "attack_bank.json"
    if bank_path.exists():
        try:
            raw = json.loads(bank_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for row in raw:
                    t = str(row.get("text") or "").strip()
                    if t and fingerprint(t) == fp:
                        atype = (row.get("attack_type") or "unknown").strip() or "unknown"
                        return {
                            "trained": True,
                            "fingerprint": fp,
                            "prompt": t,
                            "attack_type": atype,
                            "attack_display_name": _display_name(atype),
                            "label": 1,
                            "source": row.get("source") or "attack_bank",
                            "updated_at": None,
                        }
        except Exception:
            logger.warning("attack bank lookup failed", exc_info=True)

    train_path = ROOT / "data" / "processed" / "train.jsonl"
    if train_path.exists():
        try:
            with train_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = str(row.get("text") or "").strip()
                    if t and fingerprint(t) == fp:
                        atype = (row.get("attack_type") or "unknown").strip() or "unknown"
                        return {
                            "trained": True,
                            "fingerprint": fp,
                            "prompt": t,
                            "attack_type": atype,
                            "attack_display_name": _display_name(atype),
                            "label": int(row.get("label", 1)),
                            "source": row.get("source") or "train.jsonl",
                            "updated_at": None,
                        }
        except Exception:
            logger.warning("train.jsonl lookup failed", exc_info=True)

    return empty


def _display_name(attack_type: str) -> str:
    try:
        from src.layers.attack_typer import AttackTypeDetector
        return AttackTypeDetector.display_name(attack_type)
    except Exception:
        return attack_type or "unknown"
