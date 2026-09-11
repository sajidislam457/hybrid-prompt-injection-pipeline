"""
Shared anonymized malicious inbox for admin Live Lab.

Stores blocked prompts without user identity. Dedupes by fingerprint.
Statuses: new → queued (saved for train) | discarded | trained
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = ROOT / "data" / "malicious_inbox.json"
TRAINED_LOG_PATH = ROOT / "data" / "trained_log.jsonl"
_LOCK = threading.Lock()

STATUSES = ("new", "queued", "discarded", "trained")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _empty() -> Dict[str, Any]:
    return {"cases": {}, "fingerprints": {}}


def _load() -> Dict[str, Any]:
    if not STORE_PATH.exists():
        return _empty()
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data.get("cases"), dict):
            data["cases"] = {}
        if not isinstance(data.get("fingerprints"), dict):
            data["fingerprints"] = {}
        return data
    except Exception as exc:
        logger.warning("malicious inbox load failed: %s", exc)
        return _empty()


def _save(data: Dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STORE_PATH)


def compact_scenario(pipeline_result: Any) -> List[Dict[str, Any]]:
    if pipeline_result is None:
        return []
    timings = getattr(pipeline_result, "processing_time", None) or {}
    l1 = getattr(pipeline_result, "layer1", None)
    l3 = getattr(pipeline_result, "layer3", None)
    l4 = getattr(pipeline_result, "layer4", None)
    l2b = getattr(pipeline_result, "layer2b", None) or {}
    ret = getattr(pipeline_result, "retrieval", None) or {}
    return [
        {
            "id": "layer1",
            "label": "Layer 1 — Prefilter",
            "verdict": getattr(l1, "verdict", None) or getattr(l1, "action", "n/a"),
            "when_ms": round((timings.get("layer1") or 0) * 1000, 2),
        },
        {
            "id": "layer3",
            "label": "Layer 3 — Ensemble",
            "verdict": (
                "ambiguous" if getattr(l3, "is_ambiguous", False)
                else ("malicious" if getattr(l3, "final_classification", False) else "benign")
            ),
            "when_ms": round((timings.get("layer3") or 0) * 1000, 2),
        },
        {
            "id": "retrieval",
            "label": "Attack bank retrieval",
            "verdict": "hit" if ret.get("hit") else "miss",
            "when_ms": round((timings.get("retrieval") or 0) * 1000, 2),
        },
        {
            "id": "layer2b",
            "label": "Layer 2b — Semantic",
            "verdict": "malicious" if l2b.get("is_malicious") else "benign/skip",
            "when_ms": round((timings.get("layer2b") or 0) * 1000, 2),
        },
        {
            "id": "layer4",
            "label": "Layer 4 — Judge",
            "verdict": getattr(l4, "verdict", "skipped") if l4 else "skipped",
            "when_ms": round((timings.get("layer4") or 0) * 1000, 2),
        },
        {
            "id": "decision",
            "label": "Final decision",
            "verdict": getattr(pipeline_result, "action", "n/a"),
            "when_ms": round((timings.get("total") or 0) * 1000, 2),
        },
    ]


def ingest(
    prompt: str,
    *,
    attack_type: str = "unknown",
    attack_display_name: str = "Unknown",
    risk_score: float = 0.0,
    action: str = "BLOCK",
    severity: str = "high",
    decision_source: str = "unknown",
    scenario: Optional[List[Dict[str, Any]]] = None,
    timings: Optional[Dict[str, Any]] = None,
    source: str = "public_block",
    status: str = "new",
) -> Optional[Dict[str, Any]]:
    """Upsert an anonymized case. Never stores user identity."""
    prompt = (prompt or "").strip()
    if not prompt:
        return None
    fp = fingerprint(prompt)
    ts = _now()
    with _LOCK:
        data = _load()
        existing_id = data["fingerprints"].get(fp)
        if existing_id and existing_id in data["cases"]:
            case = data["cases"][existing_id]
            st = case.get("status") or "new"
            # Trained/discarded prompts that appear again are new live incidents
            if st in ("trained", "discarded"):
                fp_key = case.get("fingerprint") or fp
                del data["cases"][existing_id]
                if data["fingerprints"].get(fp_key) == existing_id:
                    del data["fingerprints"][fp_key]
                _save(data)
            else:
                case["hit_count"] = int(case.get("hit_count") or 1) + 1
                case["last_seen"] = ts
                if st == "new":
                    case["system_attack_type"] = attack_type
                    case["system_display_name"] = attack_display_name
                    case["risk_score"] = risk_score
                    case["action"] = action
                    case["severity"] = severity
                    case["decision_source"] = decision_source
                    if scenario:
                        case["scenario"] = scenario
                data["cases"][existing_id] = case
                _save(data)
                return case

        case_id = str(uuid.uuid4())
        case = {
            "id": case_id,
            "fingerprint": fp,
            "prompt": prompt,
            "prompt_preview": prompt[:220],
            "system_attack_type": attack_type,
            "system_display_name": attack_display_name,
            "risk_score": float(risk_score or 0),
            "action": action,
            "severity": severity,
            "decision_source": decision_source,
            "scenario": scenario or [],
            "timings": timings or {},
            "status": status if status in STATUSES else "new",
            "hit_count": 1,
            "first_seen": ts,
            "last_seen": ts,
            "team_attack_type": None,
            "team_label": None,
            "notes": "",
            "source": source,
            "user_identity": None,
        }
        data["cases"][case_id] = case
        data["fingerprints"][fp] = case_id
        _save(data)
        return case


def ingest_pipeline_result(prompt: str, result: Any, source: str = "public_block") -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return ingest(
        prompt,
        attack_type=getattr(result, "attack_type", None) or "unknown",
        attack_display_name=getattr(result, "attack_display_name", None) or "Unknown",
        risk_score=float(getattr(result, "final_risk_score", 0) or 0),
        action=getattr(result, "action", None) or "BLOCK",
        severity=getattr(result, "severity", None) or "high",
        decision_source=getattr(result, "decision_source", None) or "unknown",
        scenario=compact_scenario(result),
        timings=getattr(result, "processing_time", None) or {},
        source=source,
    )


def list_cases(
    status: str = "new",
    attack_type: str = "",
    q: str = "",
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    with _LOCK:
        data = _load()
        all_map = data["cases"]
        cases = list(all_map.values())
        counts = {s: 0 for s in STATUSES}
        for c in cases:
            st = c.get("status") or "new"
            if st in counts:
                counts[st] += 1
            else:
                counts["new"] += 1

    status = (status or "new").strip().lower()
    if status == "live":
        cases = [c for c in cases if (c.get("status") or "new") in ("new", "queued")]
    elif status and status != "all":
        cases = [c for c in cases if (c.get("status") or "new") == status]
    at = (attack_type or "").strip()
    if at:
        cases = [
            c for c in cases
            if (c.get("team_attack_type") or c.get("system_attack_type")) == at
        ]
    needle = (q or "").strip().lower()
    if needle:
        cases = [
            c for c in cases
            if needle in (c.get("prompt") or "").lower()
            or needle in (c.get("system_attack_type") or "")
        ]

    # Live log: newest blocked prompts first
    cases.sort(key=lambda c: c.get("last_seen") or "", reverse=True)
    total = len(cases)
    page = cases[offset: offset + max(1, min(int(limit or 50), 200))]
    summaries = [
        {
            "id": c["id"],
            "prompt": c.get("prompt") or "",
            "prompt_preview": c.get("prompt_preview") or (c.get("prompt") or "")[:220],
            "system_attack_type": c.get("system_attack_type"),
            "system_display_name": c.get("system_display_name"),
            "team_attack_type": c.get("team_attack_type"),
            "team_label": c.get("team_label"),
            "risk_score": c.get("risk_score"),
            "status": c.get("status"),
            "hit_count": c.get("hit_count"),
            "last_seen": c.get("last_seen"),
            "source": c.get("source"),
            "decision_source": c.get("decision_source"),
        }
        for c in page
    ]
    return {"items": summaries, "total": total, "offset": offset, "limit": limit, "counts": counts, "trained_file": str(TRAINED_LOG_PATH), "trained_count": count_trained_log()}


def count_trained_log() -> int:
    if not TRAINED_LOG_PATH.exists():
        return 0
    n = 0
    for line in TRAINED_LOG_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            n += 1
    return n


def archive_trained_cases() -> int:
    """Move trained rows out of the live inbox into data/trained_log.jsonl."""
    n = 0
    with _LOCK:
        data = _load()
        remove: List[tuple] = []
        TRAINED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRAINED_LOG_PATH.open("a", encoding="utf-8") as fh:
            for cid, case in data["cases"].items():
                if (case.get("status") or "") != "trained":
                    continue
                fh.write(json.dumps({
                    "archived_at": _now(),
                    "id": case.get("id"),
                    "prompt": case.get("prompt") or "",
                    "attack_type": case.get("team_attack_type") or case.get("system_attack_type"),
                    "hit_count": case.get("hit_count"),
                    "first_seen": case.get("first_seen"),
                    "last_seen": case.get("last_seen"),
                    "trained_at": case.get("trained_at"),
                    "source": case.get("source"),
                }, ensure_ascii=False) + "\n")
                remove.append((cid, case.get("fingerprint")))
                n += 1
        for cid, fp in remove:
            data["cases"].pop(cid, None)
            if fp and data["fingerprints"].get(fp) == cid:
                del data["fingerprints"][fp]
        if n:
            _save(data)
    return n


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        data = _load()
        return data["cases"].get(case_id)


def review_case(
    case_id: str,
    *,
    attack_type: str,
    label: int = 1,
    notes: str = "",
    discard: bool = False,
) -> Optional[Dict[str, Any]]:
    with _LOCK:
        data = _load()
        case = data["cases"].get(case_id)
        if not case:
            return None
        case["notes"] = (notes or "")[:500]
        case["reviewed_at"] = _now()
        if discard:
            case["status"] = "discarded"
            case["team_label"] = 0
        else:
            case["team_attack_type"] = attack_type
            case["team_label"] = int(label)
            case["status"] = "queued"
        data["cases"][case_id] = case
        _save(data)
        return case


def mark_queued_trained() -> int:
    n = 0
    with _LOCK:
        data = _load()
        for cid, case in data["cases"].items():
            if case.get("status") == "queued":
                case["status"] = "trained"
                case["trained_at"] = _now()
                data["cases"][cid] = case
                n += 1
        if n:
            _save(data)
    archive_trained_cases()
    return n
