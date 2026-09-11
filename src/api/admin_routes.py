"""
Admin-only FastAPI routes. Gated by X-Admin-Token header.
Public chatbot must never call these endpoints.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.layers.attack_typer import DISPLAY_NAMES, AttackTypeDetector
from src.utils.hw_plan import apply_eval_runtime, describe_plan

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REVIEW_QUEUE = DATA / "review_queue.jsonl"
INCOMING = DATA / "incoming"
VERSIONS = DATA / "versions"
EVALS = ROOT / "logs" / "admin_evals"
JOBS_DIR = ROOT / "logs" / "admin_jobs"
LLM_RUNTIME = ROOT / "configs" / "llm_runtime.json"
LLM_ANALYTICS = ROOT / "logs" / "llm_analytics.jsonl"
_JOB_LOG_KEEP_BYTES = 256 * 1024
_JOB_LOG_CAP_AFTER = 2 * 1024 * 1024

router = APIRouter(prefix="/admin", tags=["admin"])

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_runtime_getter = None


def _read_file_tail(path: Path, max_bytes: int = 65536) -> str:
    """Read only the end of a file — never slurp multi-GB job logs."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _cap_job_log(path: Path, keep_bytes: int = _JOB_LOG_KEEP_BYTES) -> None:
    """If a worker log blew up, keep a short tail so disk does not stay full."""
    try:
        if not path.exists() or path.stat().st_size <= _JOB_LOG_CAP_AFTER:
            return
        tail = _read_file_tail(path, keep_bytes)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            "[earlier log truncated to save disk — progress is in the .progress.json]\n" + tail,
            encoding="utf-8",
            errors="replace",
        )
        tmp.replace(path)
    except Exception:
        logger.debug("job log cap failed", exc_info=True)


def _drop_incoming_file(path: Path) -> None:
    """Staging copies in data/incoming are not needed after a successful ingest."""
    try:
        resolved = path.resolve()
        if resolved.parent.resolve() != INCOMING.resolve() or not resolved.is_file():
            return
        resolved.unlink()
    except Exception:
        logger.debug("incoming cleanup failed", exc_info=True)


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        pid_i = int(pid)
    except Exception:
        return False
    if pid_i <= 0:
        return False
    # Never shell out to tasklist here — it blocks the FastAPI event loop
    # (up to 8s) and makes Admin show API Unreachable after long evals.
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid_i
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid_i, 0)
        return True
    except OSError:
        return False


def _accuracy_job_path(job_id: str) -> Path:
    return JOBS_DIR / f"accuracy_{str(job_id)[:8]}.job.json"


def _persist_accuracy_job(job: Dict[str, Any]) -> None:
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        jid = str(job.get("id") or "")
        if not jid:
            return
        path = _accuracy_job_path(jid)
        tmp = path.with_suffix(".tmp")
        payload = {k: v for k, v in job.items() if k != "report"}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("persist accuracy job failed", exc_info=True)


def _load_accuracy_job_disk(job_id: str) -> Optional[Dict[str, Any]]:
    path = _accuracy_job_path(job_id)
    if not path.exists():
        # Allow lookup by short prefix (UI may only have first 8 in some paths).
        matches = list(JOBS_DIR.glob(f"accuracy_{str(job_id)[:8]}*.job.json"))
        if not matches:
            return None
        path = matches[0]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return raw
    except Exception:
        return None


def _accuracy_report_ok(job: Dict[str, Any]) -> bool:
    """True if the job already produced a usable report (PID-death race → not a real failure)."""
    out = job.get("out_json")
    if out and Path(str(out)).exists():
        try:
            raw = json.loads(Path(str(out)).read_text(encoding="utf-8"))
            if isinstance(raw, dict) and (raw.get("metrics") or raw.get("ablations") or raw.get("mode")):
                return True
        except Exception:
            pass
    prog = job.get("progress") or {}
    if str(prog.get("phase") or "").lower() in {"done", "finished"} and float(prog.get("pct") or 0) >= 99.0:
        return True
    if job.get("exit_code") == 0:
        return True
    # Fresh progress file on disk
    pp = job.get("progress_path")
    if pp and Path(str(pp)).exists():
        try:
            p = json.loads(Path(str(pp)).read_text(encoding="utf-8"))
            if str(p.get("phase") or "").lower() in {"done", "finished"} and float(p.get("pct") or 0) >= 99.0:
                return True
        except Exception:
            pass
    return False


def _mark_accuracy_finished_ok(job: Dict[str, Any]) -> None:
    job["status"] = "ok"
    job["error"] = None
    job["finished_at"] = job.get("finished_at") or datetime.now(timezone.utc).isoformat()
    prog = dict(job.get("progress") or {})
    prog["phase"] = prog.get("phase") or "done"
    prog["detail"] = prog.get("detail") or "Finished"
    if float(prog.get("pct") or 0) < 100:
        prog["pct"] = 100
    job["progress"] = prog


def _find_active_accuracy_job() -> Optional[Dict[str, Any]]:
    """Return a live accuracy job (memory or disk) whose worker PID is still alive."""
    with _jobs_lock:
        mem_jobs = [dict(j) for j in _jobs.values() if j.get("kind") == "accuracy"]
    for job in mem_jobs:
        if job.get("status") not in ("queued", "running"):
            continue
        pid = job.get("pid")
        if pid and _pid_alive(pid):
            return job
        if not pid and job.get("status") == "queued":
            return job
        # Stale in-memory marker — worker gone.
        with _jobs_lock:
            cur = _jobs.get(str(job.get("id") or ""))
            if cur and cur.get("status") in ("queued", "running"):
                if _accuracy_report_ok(cur):
                    _mark_accuracy_finished_ok(cur)
                else:
                    cur["status"] = "failed"
                    cur["error"] = cur.get("error") or "Worker process exited unexpectedly"
                    cur["finished_at"] = datetime.now(timezone.utc).isoformat()
                _persist_accuracy_job(cur)

    if not JOBS_DIR.exists():
        return None
    for path in sorted(JOBS_DIR.glob("accuracy_*.job.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if job.get("kind") != "accuracy":
            continue
        if job.get("status") not in ("queued", "running"):
            continue
        if job.get("pid") and _pid_alive(job.get("pid")):
            # Rehydrate into memory so polls work after API reload.
            jid = str(job.get("id") or "")
            if jid:
                with _jobs_lock:
                    _jobs[jid] = job
            return job
        # Dead PID — success if report/progress already finished (race with monitor thread).
        if _accuracy_report_ok(job):
            _mark_accuracy_finished_ok(job)
        else:
            job["status"] = "failed"
            job["error"] = job.get("error") or "Worker process exited unexpectedly"
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return None


def _kill_accuracy_worker(job: Dict[str, Any], reason: str) -> None:
    pid = job.get("pid")
    if pid:
        try:
            os.kill(int(pid), 9)
        except Exception:
            pass
    job["status"] = "failed"
    job["error"] = reason
    job["finished_at"] = datetime.now(timezone.utc).isoformat()
    _persist_accuracy_job(job)


def bind_runtime(getter) -> None:
    """Called from app.py so admin routes always see the live pipeline objects."""
    global _runtime_getter
    _runtime_getter = getter


def _admin_token() -> str:
    # Not a user password. Shared secret so public chat cannot call /admin/*.
    # Empty .env still works on localhost; set ADMIN_INTERNAL_TOKEN to override.
    return (os.getenv("ADMIN_INTERNAL_TOKEN") or "localhost-admin").strip()


def require_admin(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")) -> None:
    expected = _admin_token()
    if not expected:
        raise HTTPException(status_code=503, detail="Admin API disabled (no ADMIN_INTERNAL_TOKEN)")
    if not x_admin_token or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def get_pipeline():
    """Return live pipeline objects from the running API process."""
    if _runtime_getter is not None:
        pipeline, loaded = _runtime_getter()
        return pipeline, bool(loaded)

    # Fallback for tests / odd import orders
    import importlib
    import sys

    mod = sys.modules.get("src.api.app")
    if mod is None or not hasattr(mod, "pipeline"):
        mod = importlib.import_module("src.api.app")
    if not hasattr(mod, "pipeline"):
        raise RuntimeError("src.api.app module has no pipeline")
    return mod.pipeline, bool(getattr(mod, "pipeline_loaded", False))


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _read_jsonl(path: Path, limit: int = 500) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    buf: deque = deque(maxlen=max(1, int(limit)))
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                buf.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(buf)


def _scenario_timeline(result) -> List[Dict[str, Any]]:
    """Ordered layer story for admin Live Lab."""
    steps: List[Dict[str, Any]] = []
    timings = result.processing_time or {}

    norm = result.normalization or {}
    steps.append({
        "id": "normalize",
        "label": "Text normalization",
        "when_ms": round((timings.get("normalize") or 0) * 1000, 2),
        "verdict": "changed" if norm.get("changed") else "unchanged",
        "detail": {
            "steps": norm.get("steps") or [],
            "normalized_preview": (result.normalized_text or "")[:240],
        },
    })

    l1 = result.layer1
    steps.append({
        "id": "layer1",
        "label": "Layer 1 — Prefilter",
        "when_ms": round((timings.get("layer1") or 0) * 1000, 2),
        "verdict": getattr(l1, "verdict", None) or getattr(l1, "action", "n/a"),
        "detail": _jsonable(l1),
    })

    l2 = result.layer2 or {}
    steps.append({
        "id": "layer2",
        "label": "Layer 2 — Classical classifiers",
        "when_ms": round((timings.get("layer2") or 0) * 1000, 2),
        "verdict": "malicious" if (l2.get("is_malicious") or l2.get("prediction")) else "benign/empty",
        "detail": l2,
    })

    l3 = result.layer3
    steps.append({
        "id": "layer3",
        "label": "Layer 3 — Ensemble",
        "when_ms": round((timings.get("layer3") or 0) * 1000, 2),
        "verdict": (
            "ambiguous" if getattr(l3, "is_ambiguous", False)
            else ("malicious" if getattr(l3, "final_classification", False) else "benign")
        ),
        "detail": _jsonable(l3),
    })

    ret = result.retrieval or {}
    steps.append({
        "id": "retrieval",
        "label": "Attack bank retrieval",
        "when_ms": round((timings.get("retrieval") or 0) * 1000, 2),
        "verdict": "hit" if ret.get("hit") else "miss",
        "detail": ret,
    })

    l2b = result.layer2b or {}
    steps.append({
        "id": "layer2b",
        "label": "Layer 2b — Semantic / transformer",
        "when_ms": round((timings.get("layer2b") or 0) * 1000, 2),
        "verdict": (
            "skipped" if not l2b.get("enabled") and not timings.get("layer2b")
            else ("malicious" if l2b.get("is_malicious") else "benign")
        ),
        "detail": l2b,
    })

    l4 = result.layer4
    steps.append({
        "id": "layer4",
        "label": "Layer 4 — Ambiguity judge",
        "when_ms": round((timings.get("layer4") or 0) * 1000, 2),
        "verdict": getattr(l4, "verdict", "skipped") if l4 else "skipped",
        "detail": _jsonable(l4),
    })

    steps.append({
        "id": "decision",
        "label": "Final decision",
        "when_ms": round((timings.get("total") or 0) * 1000, 2),
        "verdict": result.action,
        "detail": {
            "is_malicious": result.is_malicious,
            "risk_score": result.final_risk_score,
            "attack_type": result.attack_type,
            "attack_display_name": result.attack_display_name,
            "decision_source": result.decision_source,
            "severity": result.severity,
        },
    })
    return steps


class ProbeRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class LabelSaveRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    attack_type: str = Field(..., min_length=1)
    label: int = Field(1, description="1=attack, 0=benign")
    notes: str = ""
    previous_attack_type: str = "unknown"
    probe_id: Optional[str] = None


class AccuracyRunRequest(BaseModel):
    mode: str = Field("heldout", pattern="^(heldout|ablation|cv)$")
    # Research default: evaluate the entire held-out test.jsonl
    full_test: bool = True
    limit: Optional[int] = Field(
        None,
        ge=0,
        le=1000000,
        description="Optional subsample; ignored when full_test=true. 0/None = all.",
    )
    rounds: int = Field(1, ge=1, le=5)
    # Only kill an in-flight run when the user explicitly forces a restart.
    force: bool = False


class AccuracyCancelRequest(BaseModel):
    job_id: Optional[str] = None


class TrainRequest(BaseModel):
    include_review_queue: bool = True
    rebuild_splits: bool = False


class DatasetIngestRequest(BaseModel):
    filename: Optional[str] = None  # file under data/incoming
    source: Optional[str] = None
    dry_run: bool = False
    async_job: bool = True  # background job (keeps API responsive)


class LlmUpdateRequest(BaseModel):
    model: Optional[str] = None
    api_key: Optional[str] = None
    provider: str = "openrouter"


@router.get("/ping")
async def admin_ping(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")):
    require_admin(x_admin_token)
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@router.get("/system")
async def admin_system(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")):
    require_admin(x_admin_token)
    try:
        pipeline, loaded = get_pipeline()
    except Exception as exc:
        logger.exception("admin_system get_pipeline failed")
        raise HTTPException(status_code=500, detail=f"pipeline bind failed: {exc}") from exc
    layer_cfg = (pipeline.config.get("layers") or {}) if pipeline else {}
    flags = (pipeline.config.get("feature_flags") or {}) if pipeline else {}
    train_file = DATA / "processed" / "train.jsonl"
    review_n = _count_jsonl_lines(REVIEW_QUEUE)
    train_cfg = (pipeline.config.get("training") or {}) if pipeline else {}
    return {
        "pipeline_loaded": loaded,
        "version": ((pipeline.config.get("system") or {}).get("version") if pipeline else None),
        "feature_flags": flags,
        "training": {
            "team_sample_weight": float(train_cfg.get("team_sample_weight", 50)),
            "team_sources": train_cfg.get("team_sources") or ["review_queue", "team_train", "inbox_review"],
        },
        "layers": {
            "layer2b": bool((layer_cfg.get("layer2b") or {}).get("enabled", True)),
            "layer4": bool((layer_cfg.get("layer4") or {}).get("enabled", True)),
            "retrieval": bool((layer_cfg.get("retrieval") or {}).get("enabled", True)),
        },
        "paths": {
            "train_exists": train_file.exists(),
            "review_queue_count": review_n,
            "incoming_count": len(list(INCOMING.glob("*"))) if INCOMING.exists() else 0,
            "model_dir": str(ROOT / "models" / "detector"),
        },
        "attack_types": [
            {"id": k, "name": v} for k, v in DISPLAY_NAMES.items()
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/lab/probe")
async def lab_probe(
    body: ProbeRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    pipeline, loaded = get_pipeline()
    if not loaded:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    prompt = body.prompt.strip()
    t0 = time.time()
    result = pipeline.process(prompt)

    inbox_case_id = None
    if result.is_malicious:
        try:
            from src.utils.malicious_inbox import ingest_pipeline_result
            case = ingest_pipeline_result(prompt, result, source="admin_probe")
            inbox_case_id = (case or {}).get("id")
        except Exception:
            logger.warning("inbox ingest from probe skipped", exc_info=True)

    # Layer 5 preview only — avoid a second full process_conversational (and double ingest)
    layer5_preview = {}
    try:
        l5 = pipeline.layer5.get_conversation_response(
            prompt, result.attack_type, result.final_risk_score
        )
        formatted = pipeline._format_layer5_result(
            l5, result.attack_type, result.final_risk_score
        )
        layer5_preview = {
            "type": formatted.get("type"),
            "response": formatted.get("response"),
            "suggestion": formatted.get("suggestion"),
            "status": formatted.get("status"),
        }
    except Exception:
        logger.warning("lab probe layer5 preview failed", exc_info=True)

    probe_id = str(uuid.uuid4())
    payload = {
        "probe_id": probe_id,
        "prompt": prompt,  # research mode A: full text, localhost-only admin
        "user_identity": None,  # explicit: never attach user details
        "inbox_case_id": inbox_case_id,
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "final": {
            "is_malicious": result.is_malicious,
            "risk_score": result.final_risk_score,
            "attack_type": result.attack_type,
            "attack_display_name": result.attack_display_name,
            "action": result.action,
            "severity": result.severity,
            "decision_source": result.decision_source,
        },
        "scenario": _scenario_timeline(result),
        "explanation": result.explanation,
        "layer5_preview": layer5_preview,
        "timings": result.processing_time,
    }
    _append_jsonl(ROOT / "logs" / "admin_probes.jsonl", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "probe_id": probe_id,
        "prompt": prompt,
        "attack_type": result.attack_type,
        "risk_score": result.final_risk_score,
        "decision_source": result.decision_source,
        "action": result.action,
        "inbox_case_id": inbox_case_id,
    })
    return payload


class InboxReviewRequest(BaseModel):
    attack_type: str = Field(..., min_length=1)
    label: int = 1
    notes: str = ""
    discard: bool = False


class InboxManualRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    attack_type: str = Field(..., min_length=1)
    label: int = 1
    notes: str = ""


class LabLookupRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


@router.post("/lab/lookup")
async def lab_lookup(
    body: LabLookupRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    """Check if a prompt is already in team overrides / attack bank / train.jsonl."""
    require_admin(x_admin_token)
    from src.utils.team_overrides import lookup_trained
    return lookup_trained(body.prompt)


@router.get("/inbox")
async def inbox_list(
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    status: str = "live",
    attack_type: str = "",
    q: str = "",
    limit: int = 200,
    offset: int = 0,
):
    require_admin(x_admin_token)
    from src.utils.malicious_inbox import archive_trained_cases, list_cases
    # Clean up any trained rows left in the inbox file from older builds
    if status == "live":
        archive_trained_cases()
    return list_cases(status=status, attack_type=attack_type, q=q, limit=limit, offset=offset)


@router.post("/inbox/manual")
async def inbox_manual(
    body: InboxManualRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    from src.utils.malicious_inbox import get_case, ingest, review_case
    pipeline, loaded = get_pipeline()
    prompt = body.prompt.strip()
    scenario = []
    timings = {}
    sys_type = body.attack_type.strip()
    display = AttackTypeDetector.display_name(sys_type)
    risk = 0.9 if int(body.label) == 1 else 0.0
    if loaded and pipeline:
        try:
            result = pipeline.process(prompt)
            from src.utils.malicious_inbox import compact_scenario
            scenario = compact_scenario(result)
            timings = result.processing_time or {}
            if int(body.label) == 1:
                sys_type = body.attack_type.strip() or result.attack_type
                display = AttackTypeDetector.display_name(sys_type)
                risk = max(float(result.final_risk_score or 0), 0.75)
        except Exception:
            logger.warning("manual inbox process failed", exc_info=True)
    case = ingest(
        prompt,
        attack_type=sys_type,
        attack_display_name=display,
        risk_score=risk,
        action="BLOCK" if int(body.label) == 1 else "ALLOW",
        decision_source="admin_manual",
        scenario=scenario,
        timings=timings,
        source="manual",
        status="queued",
    )
    row = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "attack_type": body.attack_type.strip(),
        "attack_display_name": AttackTypeDetector.display_name(body.attack_type.strip()),
        "label": int(body.label),
        "notes": (body.notes or "")[:500],
        "previous_attack_type": "manual",
        "case_id": (case or {}).get("id"),
        "source": "inbox_manual",
    }
    _append_jsonl(REVIEW_QUEUE, row)
    if case and case.get("id"):
        review_case(
            case["id"],
            attack_type=body.attack_type.strip(),
            label=int(body.label),
            notes=body.notes,
            discard=False,
        )
        case = get_case(case["id"])
    return {"ok": True, "item": case}


@router.get("/inbox/{case_id}")
async def inbox_get(
    case_id: str,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    from src.utils.malicious_inbox import get_case
    case = get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.post("/inbox/{case_id}/review")
async def inbox_review(
    case_id: str,
    body: InboxReviewRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    from src.utils.malicious_inbox import review_case
    case = review_case(
        case_id,
        attack_type=body.attack_type.strip(),
        label=int(body.label),
        notes=body.notes,
        discard=body.discard,
    )
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if not body.discard:
        row = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": case.get("prompt") or "",
            "attack_type": body.attack_type.strip(),
            "attack_display_name": AttackTypeDetector.display_name(body.attack_type.strip()),
            "label": int(body.label),
            "notes": (body.notes or "")[:500],
            "previous_attack_type": case.get("system_attack_type") or "unknown",
            "case_id": case_id,
            "source": "inbox_review",
        }
        _append_jsonl(REVIEW_QUEUE, row)
    return {"ok": True, "item": case}



@router.get("/labels")
async def list_labels(
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    limit: int = 200,
):
    require_admin(x_admin_token)
    return {"items": list(reversed(_read_jsonl(REVIEW_QUEUE, limit=limit)))}


@router.post("/labels")
async def save_label(
    body: LabelSaveRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    atype = body.attack_type.strip()
    if atype not in DISPLAY_NAMES and atype != "benign":
        # allow custom research labels but prefer known set
        pass
    row = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": body.prompt.strip(),
        "attack_type": atype,
        "attack_display_name": AttackTypeDetector.display_name(atype),
        "label": int(body.label),
        "notes": (body.notes or "")[:500],
        "previous_attack_type": body.previous_attack_type,
        "probe_id": body.probe_id,
        # Privacy: no user_id / conversation_id / IP
        "source": "admin_label_studio",
    }
    _append_jsonl(REVIEW_QUEUE, row)
    return {"ok": True, "item": row}


@router.post("/accuracy/run")
async def accuracy_run(
    body: AccuracyRunRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    EVALS.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    # Only one accuracy job at a time. Do NOT kill on accidental re-click —
    # that was the main reason heldout/ablation looked "failed".
    active = _find_active_accuracy_job()
    if active and not body.force:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "An accuracy evaluation is already running. "
                    "Wait for it to finish, or Cancel it first."
                ),
                "job_id": active.get("id"),
                "mode": active.get("mode"),
                "status": active.get("status"),
                "progress": active.get("progress"),
                "pid": active.get("pid"),
            },
        )
    if active and body.force:
        with _jobs_lock:
            jid = str(active.get("id") or "")
            if jid and jid in _jobs:
                _kill_accuracy_worker(_jobs[jid], "Cancelled to start a new accuracy run")
            else:
                _kill_accuracy_worker(active, "Cancelled to start a new accuracy run")

    job_id = str(uuid.uuid4())
    out_json = EVALS / f"{body.mode}_{job_id[:8]}.json"
    out_md = EVALS / f"{body.mode}_{job_id[:8]}.md"
    log_path = JOBS_DIR / f"accuracy_{job_id[:8]}.log"
    progress_path = JOBS_DIR / f"accuracy_{job_id[:8]}.progress.json"

    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(os.environ.get("PYTHON", "python"))

    # Research default: full held-out test.jsonl (Check_Accuracy --limit 0 = all)
    eval_limit = 0 if body.full_test or body.limit in (None, 0) else int(body.limit)

    cmd = [
        str(py),
        "-u",  # unbuffered
        str(ROOT / "scripts" / "Check_Accuracy.py"),
        "--mode", body.mode,
        "--limit", str(eval_limit),
        "--no-balance",  # full file as-is for paper integrity when limit=0
    ]
    # When subsampling (limit>0), keep balance for fair Acc; when full set, no-balance is fine.
    if eval_limit > 0:
        cmd = [c for c in cmd if c != "--no-balance"]

    env = os.environ.copy()
    env["ADMIN_ACCURACY_OUT_JSON"] = str(out_json)
    env["ADMIN_ACCURACY_OUT_MD"] = str(out_md)
    env["ADMIN_ACCURACY_PROGRESS"] = str(progress_path)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # Bulk eval disables decision I/O in Check_Accuracy; also silence warning spam in the job log.
    env["PYTHONWARNINGS"] = "ignore"
    env["ACCURACY_DISABLE_DECISION_LOG"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TRANSFORMERS_VERBOSITY"] = "error"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Pick GPU vs CPU and a thread budget that leaves cores for Admin.
    hw_plan = apply_eval_runtime(env)
    logger.info(
        "Accuracy worker hardware path=%s device=%s omp=%s (%s)",
        hw_plan.get("path"),
        hw_plan.get("layer2b_device"),
        hw_plan.get("omp_threads"),
        hw_plan.get("reason"),
    )

    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": "accuracy",
            "mode": body.mode,
            "limit": eval_limit,
            "full_test": bool(body.full_test or eval_limit == 0),
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "log_path": str(log_path),
            "progress_path": str(progress_path),
            "out_json": str(out_json),
            "out_md": str(out_md),
            "hw_plan": hw_plan,
            "progress": {
                "phase": "queued",
                "done": 0,
                "total": 0,
                "pct": 0,
                "detail": f"Queued · {describe_plan(hw_plan)}",
                "hw_path": hw_plan.get("path"),
                "hw_reason": hw_plan.get("reason"),
                "hw_device": hw_plan.get("layer2b_device"),
                "hw_threads": hw_plan.get("omp_threads"),
            },
        }
        _persist_accuracy_job(_jobs[job_id])

    def _read_progress_file() -> Dict[str, Any]:
        if not progress_path.exists():
            return {}
        try:
            return json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _parse_progress_from_log(text: str) -> Dict[str, Any]:
        import re as _re
        progress: Dict[str, Any] = {
            "phase": "running",
            "done": 0,
            "total": 0,
            "pct": 0,
            "detail": "Starting…",
            "ablation": None,
        }
        if not text:
            return progress
        for ln in text.splitlines():
            m = _re.search(r"Held-out samples:\s*(\d+)", ln)
            if m and progress["total"] <= 0:
                progress["total"] = int(m.group(1))
            m = _re.search(r"Ablation on\s+(\d+)\s+held-out", ln)
            if m:
                progress["total"] = int(m.group(1))
            m = _re.search(r"===\s*Ablation:\s*(\w+)\s*===", ln)
            if m:
                progress["ablation"] = m.group(1)
            m = _re.search(r"samples\s+(\d+)\s*/\s*(\d+)", ln)
            if m:
                progress["done"] = int(m.group(1))
                progress["total"] = int(m.group(2))
        if progress["total"] > 0:
            progress["pct"] = round(100.0 * progress["done"] / progress["total"], 1)
        if progress["ablation"]:
            progress["phase"] = "ablation"
            progress["detail"] = (
                f"Ablation `{progress['ablation']}` · {progress['done']}/{progress['total'] or '?'}"
            )
        elif progress["done"] or progress["total"]:
            progress["phase"] = "scoring"
            progress["detail"] = f"Scoring samples {progress['done']}/{progress['total'] or '?'}"
        elif "Loading pipeline" in text:
            progress["phase"] = "loading"
            progress["detail"] = "Loading pipeline / models…"
        return progress

    def _run():
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
            _jobs[job_id]["progress"] = {
                "phase": "loading",
                "done": 0,
                "total": 0,
                "pct": 0,
                "detail": "Loading pipeline…",
            }
            _persist_accuracy_job(_jobs[job_id])
        logf = None
        proc = None
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered log; child owns the handle. Parent only re-opens for reads.
            logf = open(log_path, "w", encoding="utf-8", buffering=1, errors="replace")
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
            )
            with _jobs_lock:
                _jobs[job_id]["pid"] = proc.pid
                _persist_accuracy_job(_jobs[job_id])

            # No hard wall for huge test sets (100k–900k). Soft heartbeat only.
            started = time.time()
            last_done = -1
            stall_started = time.time()
            last_persist = 0.0
            while True:
                code = proc.poll()
                if code is not None:
                    break
                time.sleep(2)
                prog = _read_progress_file()
                if not prog:
                    try:
                        prog = _parse_progress_from_log(_read_file_tail(log_path, 65536))
                    except Exception:
                        prog = {}
                if prog:
                    with _jobs_lock:
                        if _jobs.get(job_id):
                            _jobs[job_id]["progress"] = {
                                "phase": prog.get("phase") or "scoring",
                                "done": int(prog.get("done") or 0),
                                "total": int(prog.get("total") or 0),
                                "pct": float(prog.get("pct") or 0),
                                "detail": prog.get("detail")
                                or f"Scoring {prog.get('done', 0)}/{prog.get('total', '?')}",
                                "ablation": prog.get("ablation"),
                                "errors": prog.get("errors"),
                            }
                            # Persist every ~15s so API reloads can resume polling.
                            now = time.time()
                            if now - last_persist >= 15:
                                _persist_accuracy_job(_jobs[job_id])
                                last_persist = now
                    done_now = int(prog.get("done") or 0)
                    if done_now != last_done:
                        last_done = done_now
                        stall_started = time.time()
                    # If no progress for 30 minutes after scoring started, mark stalled (still wait).
                    elif last_done >= 0 and (time.time() - stall_started) > 1800:
                        with _jobs_lock:
                            if _jobs.get(job_id):
                                cur = dict(_jobs[job_id].get("progress") or {})
                                cur["detail"] = (
                                    f"Still running (no new samples for "
                                    f"{int((time.time() - stall_started) / 60)}m) · "
                                    f"{cur.get('done', 0)}/{cur.get('total', '?')}"
                                )
                                _jobs[job_id]["progress"] = cur

            if logf:
                try:
                    logf.flush()
                    logf.close()
                except Exception:
                    pass
                logf = None

            _cap_job_log(log_path)
            status = "ok" if code == 0 else "failed"
            prog = _read_progress_file() or _parse_progress_from_log(
                _read_file_tail(log_path, 65536) if log_path.exists() else ""
            )
            if status == "ok":
                prog = {
                    **prog,
                    "pct": 100,
                    "phase": "done",
                    "detail": "Finished",
                    "done": prog.get("done") or prog.get("total") or 0,
                    "total": prog.get("total") or prog.get("done") or 0,
                }
            with _jobs_lock:
                _jobs[job_id]["status"] = status
                _jobs[job_id]["exit_code"] = code
                _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                _jobs[job_id]["log_path"] = str(log_path)
                _jobs[job_id]["out_json"] = str(out_json) if out_json.exists() else None
                _jobs[job_id]["progress"] = prog
                if status == "ok":
                    # Clear any false "exited unexpectedly" set by a PID race.
                    _jobs[job_id]["error"] = None
                elif not _jobs[job_id].get("error"):
                    try:
                        tail = _read_file_tail(log_path, 2000)
                        _jobs[job_id]["error"] = tail.strip().splitlines()[-1] if tail.strip() else f"exit {code}"
                    except Exception:
                        _jobs[job_id]["error"] = f"exit {code}"
                _persist_accuracy_job(_jobs[job_id])
            elapsed = time.time() - started
            logger.info("Accuracy job %s finished status=%s elapsed=%.1fs", job_id[:8], status, elapsed)
        except Exception as exc:
            if proc and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            if logf:
                try:
                    logf.close()
                except Exception:
                    pass
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(exc)
                _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                _jobs[job_id]["progress"] = {
                    "phase": "failed",
                    "done": 0,
                    "total": 0,
                    "pct": 0,
                    "detail": str(exc),
                }
                _persist_accuracy_job(_jobs[job_id])

    # Non-daemon so a brief GC/reload is less likely to silently abandon work mid-run.
    threading.Thread(target=_run, daemon=False).start()
    return {"job_id": job_id, "status": "queued"}


@router.get("/accuracy/active")
async def accuracy_active(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")):
    require_admin(x_admin_token)
    job = _find_active_accuracy_job()
    if not job:
        return {"active": False, "job": None}
    # Refresh progress from disk for UI.
    progress_path = Path(job.get("progress_path") or "")
    if progress_path.exists():
        try:
            prog = json.loads(progress_path.read_text(encoding="utf-8"))
            job = dict(job)
            job["progress"] = {
                "phase": prog.get("phase") or "scoring",
                "done": int(prog.get("done") or 0),
                "total": int(prog.get("total") or 0),
                "pct": float(prog.get("pct") or 0),
                "detail": prog.get("detail")
                or f"Scoring {prog.get('done', 0)}/{prog.get('total', '?')}",
                "ablation": prog.get("ablation"),
                "errors": prog.get("errors"),
            }
        except Exception:
            pass
    return {"active": True, "job": job}


@router.post("/accuracy/cancel")
async def accuracy_cancel(
    body: AccuracyCancelRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    job = None
    if body.job_id:
        with _jobs_lock:
            job = _jobs.get(body.job_id)
        if not job:
            job = _load_accuracy_job_disk(body.job_id)
    if not job:
        job = _find_active_accuracy_job()
    if not job:
        return {"cancelled": False, "detail": "No active accuracy job"}
    with _jobs_lock:
        jid = str(job.get("id") or "")
        if jid and jid in _jobs:
            _kill_accuracy_worker(_jobs[jid], "Cancelled by user")
            out = dict(_jobs[jid])
        else:
            _kill_accuracy_worker(job, "Cancelled by user")
            out = dict(job)
            out["status"] = "failed"
            out["error"] = "Cancelled by user"
    return {"cancelled": True, "job": out}


@router.get("/accuracy/jobs/{job_id}")
async def accuracy_job(
    job_id: str,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        job = _load_accuracy_job_disk(job_id)
        if job:
            with _jobs_lock:
                _jobs[job_id] = job
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = dict(job)
    progress_path = Path(job.get("progress_path") or (JOBS_DIR / f"accuracy_{job_id[:8]}.progress.json"))
    log_path = Path(job.get("log_path") or (JOBS_DIR / f"accuracy_{job_id[:8]}.log"))
    if job.get("status") in ("queued", "running"):
        # If worker died, surface failed instead of forever-running —
        # unless the report/progress already shows a clean finish (PID race).
        if job.get("pid") and not _pid_alive(job.get("pid")):
            if _accuracy_report_ok(result):
                _mark_accuracy_finished_ok(result)
                with _jobs_lock:
                    if job_id in _jobs:
                        _mark_accuracy_finished_ok(_jobs[job_id])
                        _persist_accuracy_job(_jobs[job_id])
            else:
                result["status"] = "failed"
                result["error"] = result.get("error") or "Worker process exited unexpectedly"
                result["finished_at"] = result.get("finished_at") or datetime.now(timezone.utc).isoformat()
                with _jobs_lock:
                    if job_id in _jobs:
                        _jobs[job_id].update({
                            "status": "failed",
                            "error": result["error"],
                            "finished_at": result["finished_at"],
                        })
                        _persist_accuracy_job(_jobs[job_id])
        elif progress_path.exists():
            try:
                prog = json.loads(progress_path.read_text(encoding="utf-8"))
                result["progress"] = {
                    "phase": prog.get("phase") or "scoring",
                    "done": int(prog.get("done") or 0),
                    "total": int(prog.get("total") or 0),
                    "pct": float(prog.get("pct") or 0),
                    "detail": prog.get("detail")
                    or f"Scoring {prog.get('done', 0)}/{prog.get('total', '?')}",
                    "ablation": prog.get("ablation"),
                    "errors": prog.get("errors"),
                }
            except Exception:
                pass
    if log_path.exists():
        try:
            result["log_tail"] = _read_file_tail(log_path, 8000)
        except Exception:
            pass
    out = job.get("out_json")
    if out and Path(out).exists():
        try:
            result["report"] = json.loads(Path(out).read_text(encoding="utf-8"))
        except Exception:
            pass
    return result


@router.get("/accuracy/reports")
async def accuracy_reports(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")):
    require_admin(x_admin_token)
    EVALS.mkdir(parents=True, exist_ok=True)
    files = sorted(EVALS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]
    # Also surface classic logs/ reports if present
    logs_dir = ROOT / "logs"
    extra = []
    if logs_dir.exists():
        for name in ("check_accuracy_heldout.json", "check_accuracy_ablation.json"):
            p = logs_dir / name
            if p.exists():
                extra.append(p)
    seen = set()
    ordered: List[Path] = []
    for f in list(files) + extra:
        key = str(f.resolve())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(f)
    # Prefer admin_evals over classic logs/check_accuracy_*.json when mtimes match.
    ordered = sorted(
        ordered,
        key=lambda p: (p.stat().st_mtime, 1 if p.parent.resolve() == EVALS.resolve() else 0),
        reverse=True,
    )[:40]

    items = []
    # One eval run writes both logs/check_accuracy_*.json and logs/admin_evals/<job>.json.
    # Prefer the admin_evals copy; drop classic duplicates with the same generated_at + metrics.
    seen_content: set = set()
    for f in ordered:
        summary: Dict[str, Any] = {}
        mode = ""
        title = f.name
        generated_at = ""
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            mode = str(raw.get("mode") or "")
            title = str(raw.get("title") or f.name)
            generated_at = str(raw.get("generated_at") or "")
            if isinstance(raw.get("metrics"), dict):
                m = raw["metrics"]
                summary = {
                    "n": m.get("n"),
                    "accuracy": m.get("accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1": m.get("f1"),
                    "auc_roc": m.get("auc_roc"),
                    "fpr": m.get("fpr"),
                }
            elif isinstance(raw.get("aggregate"), dict):
                agg = raw["aggregate"]
                summary = {
                    "accuracy": (agg.get("accuracy") or {}).get("mean"),
                    "precision": (agg.get("precision") or {}).get("mean"),
                    "recall": (agg.get("recall") or {}).get("mean"),
                    "f1": (agg.get("f1") or {}).get("mean"),
                    "fpr": (agg.get("fpr") or {}).get("mean"),
                }
            elif isinstance(raw.get("ablations"), dict):
                full = (raw["ablations"].get("full") or {}).get("metrics") or {}
                summary = {
                    "n": raw.get("n"),
                    "accuracy": full.get("accuracy"),
                    "precision": full.get("precision"),
                    "recall": full.get("recall"),
                    "f1": full.get("f1"),
                    "auc_roc": full.get("auc_roc"),
                    "fpr": full.get("fpr"),
                    "ablation_count": len(raw["ablations"]),
                }
        except Exception:
            pass

        is_classic = f.parent.resolve() == logs_dir.resolve() and f.name.startswith("check_accuracy_")
        content_key = (
            generated_at,
            mode,
            summary.get("n"),
            summary.get("f1"),
            summary.get("recall"),
            summary.get("fpr"),
            summary.get("ablation_count"),
        )
        if generated_at and content_key in seen_content:
            continue
        # If classic appears before its admin_evals twin (same mtime), skip classic when a twin exists.
        if is_classic and generated_at:
            has_admin_twin = False
            for p in ordered:
                if p.parent.resolve() != EVALS.resolve():
                    continue
                try:
                    other = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(other.get("generated_at") or "") == generated_at and str(other.get("mode") or "") == mode:
                    has_admin_twin = True
                    break
            if has_admin_twin:
                continue

        if generated_at:
            seen_content.add(content_key)
        items.append({
            "name": f.name,
            "path": str(f),
            "mtime": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(),
            "size": f.stat().st_size,
            "mode": mode,
            "title": title,
            "summary": summary,
            "paper_dir": str((f.parent / "paper" / f.stem)) if (f.parent / "paper" / f.stem).exists() else None,
        })
    return {"items": items}


@router.get("/accuracy/reports/{name}")
async def accuracy_report_detail(
    name: str,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    safe = Path(name).name
    if not safe.endswith(".json"):
        raise HTTPException(status_code=400, detail="Report name must end with .json")
    candidates = [EVALS / safe, ROOT / "logs" / safe]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        raise HTTPException(status_code=404, detail="Report not found")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read report: {exc}") from exc
    paper = path.parent / "paper" / path.stem
    return {
        "name": path.name,
        "path": str(path),
        "paper_dir": str(paper) if paper.exists() else None,
        "report": report,
    }


@router.get("/datasets/taxonomy")
async def datasets_taxonomy(
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    from src.utils.dataset_ingest import get_taxonomy, list_incoming_files

    tax = get_taxonomy()
    train_n = 0
    train_file = DATA / "processed" / "train.jsonl"
    if train_file.exists():
        # Fast line count (no JSON parse) so status panel stays snappy.
        with train_file.open("rb") as fh:
            train_n = sum(1 for _ in fh)
    return {
        **tax,
        "incoming": list_incoming_files(),
        "train_rows": train_n,
    }


@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    """Save file to incoming only (fast). Heavy normalize/dedupe runs via /datasets/ingest job."""
    require_admin(x_admin_token)
    from src.utils.dataset_ingest import _source_from_filename

    INCOMING.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "upload.jsonl").name
    if not name.endswith((".jsonl", ".json", ".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Allowed: .jsonl .json .csv .txt")
    dest = INCOMING / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
    content = await file.read()
    dest.write_bytes(content)

    preview: Dict[str, Any] = {"lines": 0, "sample_keys": [], "samples": []}
    try:
        text = content.decode("utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        preview["lines"] = len(lines)
        for ln in lines[:3]:
            if ln.startswith("{"):
                obj = json.loads(ln)
                preview["sample_keys"] = list(obj.keys())
                preview["samples"].append({
                    "text": str(obj.get("text") or obj.get("payload") or obj.get("prompt") or "")[:120],
                    "label": obj.get("label"),
                    "attack_type": obj.get("attack_type") or obj.get("attack_category") or obj.get("category"),
                })
    except Exception as exc:
        preview["parse_warning"] = str(exc)

    return {
        "ok": True,
        "path": str(dest),
        "filename": dest.name,
        "source_guess": _source_from_filename(name),
        "preview": preview,
    }


@router.post("/datasets/accept")
async def accept_incoming(
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    """Copy incoming files into data/raw for archival / processing."""
    require_admin(x_admin_token)
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    INCOMING.mkdir(parents=True, exist_ok=True)
    moved = []
    for f in INCOMING.iterdir():
        if f.is_file():
            dest = raw / f.name
            shutil.move(str(f), dest)
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass
            moved.append(str(dest))
    return {"ok": True, "moved": moved}


@router.post("/datasets/ingest")
async def ingest_dataset(
    body: DatasetIngestRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    """Normalize → map → dedupe → append. Default: background job so /health stays responsive."""
    require_admin(x_admin_token)
    from src.utils.dataset_ingest import ingest_file_into_train, list_incoming_files, preview_ingest

    INCOMING.mkdir(parents=True, exist_ok=True)
    files = list_incoming_files()
    if not files:
        raise HTTPException(status_code=400, detail="No files in data/incoming — upload first")

    target_name = (body.filename or "").strip()
    if target_name:
        path = INCOMING / Path(target_name).name
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not in incoming: {target_name}")
    else:
        path = Path(files[0]["path"])

    if body.dry_run:
        return preview_ingest(path, source=body.source)

    # Sync path (tests / small files)
    if not body.async_job:
        result = ingest_file_into_train(path, source=body.source, dry_run=False)
        _drop_incoming_file(path)
        train_n = 0
        train_file = DATA / "processed" / "train.jsonl"
        if train_file.exists():
            with train_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        train_n += 1
        result["train_rows"] = train_n
        return result

    job_id = str(uuid.uuid4())
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def _run():
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
            _jobs[job_id]["progress"] = {"stage": "starting", "detail": "Queued ingest…"}

        def on_progress(stage: str, detail: str, extra: Dict[str, Any]) -> None:
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["progress"] = {
                        "stage": stage,
                        "detail": detail,
                        **(extra or {}),
                    }

        try:
            result = ingest_file_into_train(
                path,
                source=body.source,
                dry_run=False,
                progress_cb=on_progress,
            )
            _drop_incoming_file(path)
            train_n = 0
            train_file = DATA / "processed" / "train.jsonl"
            if train_file.exists():
                with train_file.open("rb") as fh:
                    train_n = sum(1 for _ in fh)
            result["train_rows"] = train_n
            with _jobs_lock:
                _jobs[job_id]["status"] = "ok"
                _jobs[job_id]["result"] = result
                _jobs[job_id]["appended"] = result.get("appended")
                _jobs[job_id]["train_rows"] = train_n
                _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                _jobs[job_id]["progress"] = {
                    "stage": "done",
                    "detail": f"Added {result.get('appended', 0)} rows",
                }
        except Exception as exc:
            logger.exception("ingest job failed")
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(exc)
                _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                _jobs[job_id]["progress"] = {"stage": "failed", "detail": str(exc)}

    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": "ingest",
            "status": "queued",
            "filename": path.name,
            "source": body.source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "progress": {"stage": "queued", "detail": "Waiting…"},
        }
    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "job_id": job_id, "status": "queued", "filename": path.name}


@router.post("/datasets/train")
async def train_model(
    body: TrainRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    log_path = JOBS_DIR / f"train_{job_id[:8]}.log"

    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path("python")

    def _merge_review_into_attack_bank() -> int:
        if not body.include_review_queue:
            return 0
        bank_path = DATA / "attack_bank.json"
        items = _read_jsonl(REVIEW_QUEUE, limit=100000)
        if not items:
            return 0
        from src.utils.malicious_inbox import fingerprint
        bank: List[Dict[str, Any]] = []
        if bank_path.exists():
            try:
                raw = json.loads(bank_path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    bank = raw
            except Exception:
                bank = []
        seen: Dict[str, int] = {}
        for i, row in enumerate(bank):
            fp = fingerprint(str(row.get("text") or ""))
            if fp:
                seen[fp] = i
        n = 0
        for it in items:
            if int(it.get("label", 1)) != 1:
                continue
            text = (it.get("prompt") or it.get("text") or "").strip()
            if not text:
                continue
            atype = it.get("attack_type") or "unknown"
            fp = fingerprint(text)
            entry = {"text": text, "attack_type": atype, "source": "team_train"}
            if fp in seen:
                bank[seen[fp]] = entry
                n += 1
            else:
                bank.append(entry)
                seen[fp] = len(bank) - 1
                n += 1
        if n:
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            bank_path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
        return n

    def _merge_review_into_train():
        """Upsert review items into train.jsonl by fingerprint (no duplicate over-weight)."""
        if not body.include_review_queue:
            return 0
        from src.utils.malicious_inbox import fingerprint
        processed = DATA / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        train_file = processed / "train.jsonl"
        items = _read_jsonl(REVIEW_QUEUE, limit=100000)
        if not items:
            return 0

        # Load full train set (not _read_jsonl limit-tail) so rewrite cannot drop rows.
        existing: List[Dict[str, Any]] = []
        if train_file.exists():
            with train_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        rows: List[Dict[str, Any]] = []
        seen: Dict[str, int] = {}
        for row in existing:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            fp = fingerprint(text)
            row = {**row, "text": text}
            if fp in seen:
                rows[seen[fp]] = row
            else:
                seen[fp] = len(rows)
                rows.append(row)

        n = 0
        for it in items:
            text = (it.get("prompt") or it.get("text") or "").strip()
            if not text:
                continue
            fp = fingerprint(text)
            entry = {
                "text": text,
                "label": int(it.get("label", 1)),
                "attack_category": it.get("attack_type") or it.get("attack_category") or "unknown",
                "attack_type": it.get("attack_type") or it.get("attack_category") or "unknown",
                "source": (it.get("source") or "review_queue").strip(),
                "id": it.get("id"),
            }
            if fp in seen:
                rows[seen[fp]] = entry
            else:
                seen[fp] = len(rows)
                rows.append(entry)
            n += 1

        if n:
            tmp = train_file.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp.replace(train_file)
        return n

    def _run():
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
        try:
            merged = _merge_review_into_train()
            override_n = 0
            if body.include_review_queue:
                try:
                    from src.utils.team_overrides import apply_review_items
                    override_n = apply_review_items(_read_jsonl(REVIEW_QUEUE, limit=100000))
                except Exception:
                    logger.warning("team overrides apply failed", exc_info=True)
            with _jobs_lock:
                _jobs[job_id]["merged_review"] = merged
                _jobs[job_id]["team_overrides"] = override_n
                try:
                    from src.utils.helpers import load_config
                    tcfg = (load_config(ROOT / "configs" / "config.yaml") or {}).get("training") or {}
                    _jobs[job_id]["team_sample_weight"] = float(tcfg.get("team_sample_weight", 50))
                except Exception:
                    _jobs[job_id]["team_sample_weight"] = 50
            with log_path.open("w", encoding="utf-8") as logf:
                steps = []
                if body.rebuild_splits:
                    steps.append(["--step", "process"])
                steps.append(["--step", "train"])
                code = 0
                for args in steps:
                    proc = subprocess.run(
                        [str(py), str(ROOT / "main.py"), *args],
                        cwd=str(ROOT),
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        timeout=7200,
                    )
                    code = proc.returncode
                    if code != 0:
                        break
            bank_rebuilt = False
            bank_added = 0
            if code == 0:
                with log_path.open("a", encoding="utf-8") as logf:
                    logf.write("\n=== Rebuild attack_bank.json from train.jsonl (val/test excluded) ===\n")
                    logf.flush()
                    bank_proc = subprocess.run(
                        [str(py), str(ROOT / "scripts" / "build_attack_bank.py")],
                        cwd=str(ROOT),
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                    )
                    bank_rebuilt = bank_proc.returncode == 0
                    if not bank_rebuilt:
                        logf.write(f"\n[warn] attack bank rebuild exit {bank_proc.returncode}\n")
                try:
                    bank_added = _merge_review_into_attack_bank()
                except Exception:
                    logger.warning("review→attack bank merge failed", exc_info=True)
            with _jobs_lock:
                _jobs[job_id]["status"] = "ok" if code == 0 else "failed"
                _jobs[job_id]["exit_code"] = code
                _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
                _jobs[job_id]["log_path"] = str(log_path)
                _jobs[job_id]["attack_bank_rebuilt"] = bank_rebuilt
                _jobs[job_id]["attack_bank_added"] = bank_added
            if code == 0 and body.include_review_queue:
                try:
                    from src.utils.malicious_inbox import mark_queued_trained
                    marked = mark_queued_trained()
                    with _jobs_lock:
                        _jobs[job_id]["inbox_marked_trained"] = marked
                except Exception:
                    logger.warning("inbox trained-mark failed", exc_info=True)
                try:
                    # Archive consumed queue so the next train does not re-merge duplicates
                    if REVIEW_QUEUE.exists() and REVIEW_QUEUE.stat().st_size > 0:
                        archive = DATA / "versions" / f"review_queue_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(REVIEW_QUEUE), str(archive))
                        with _jobs_lock:
                            _jobs[job_id]["review_queue_archived"] = str(archive)
                except Exception:
                    logger.warning("review queue archive failed", exc_info=True)
            if code == 0:
                # Hot-reload Layer 2 + attack bank into the running API
                try:
                    pipeline, loaded = get_pipeline()
                    reloaded = bool(loaded and pipeline and pipeline.load_models())
                    bank_reloaded = False
                    if loaded and pipeline and getattr(pipeline, "retriever", None):
                        try:
                            pipeline.retriever._build()
                            bank_reloaded = True
                        except Exception:
                            logger.warning("attack bank reload failed", exc_info=True)
                    if loaded and pipeline:
                        try:
                            pipeline._reload_team_overrides()
                        except Exception:
                            logger.warning("team overrides reload failed", exc_info=True)
                    with _jobs_lock:
                        _jobs[job_id]["model_reloaded"] = reloaded
                        _jobs[job_id]["attack_bank_reloaded"] = bank_reloaded
                    if reloaded:
                        logger.info("Live model reloaded after train job %s", job_id[:8])
                    else:
                        logger.warning("Train ok but model reload failed for job %s", job_id[:8])
                except Exception:
                    logger.warning("model hot-reload failed", exc_info=True)
                    with _jobs_lock:
                        _jobs[job_id]["model_reloaded"] = False
        except Exception as exc:
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(exc)
                _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": "train",
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = dict(job)
    log_path = job.get("log_path")
    if log_path and Path(log_path).exists():
        try:
            result["log_tail"] = Path(log_path).read_text(encoding="utf-8")[-8000:]
        except Exception:
            pass
    return result


def _mask_key(key: str) -> str:
    key = key or ""
    if len(key) < 8:
        return "••••" if key else ""
    return key[:4] + "••••" + key[-4:]


@router.get("/llm")
async def get_llm(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")):
    require_admin(x_admin_token)
    cfg = {
        "provider": "openrouter",
        "model": os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini",
        "api_key_set": bool(os.getenv("OPENROUTER_API_KEY")),
        "api_key_masked": _mask_key(os.getenv("OPENROUTER_API_KEY") or ""),
    }
    if LLM_RUNTIME.exists():
        try:
            disk = json.loads(LLM_RUNTIME.read_text(encoding="utf-8"))
            cfg["provider"] = disk.get("provider") or cfg["provider"]
            cfg["model"] = disk.get("model") or cfg["model"]
            if disk.get("api_key"):
                cfg["api_key_set"] = True
                cfg["api_key_masked"] = _mask_key(disk["api_key"])
        except Exception:
            pass
    analytics = _read_jsonl(LLM_ANALYTICS, limit=500)
    by_model: Dict[str, Dict[str, Any]] = {}
    for row in analytics:
        m = row.get("model") or "unknown"
        slot = by_model.setdefault(m, {
            "model": m,
            "events": 0,
            "switches": 0,
            "probes": 0,
            "errors": 0,
            "latency_sum": 0.0,
            "latency_n": 0,
        })
        slot["events"] += 1
        kind = row.get("kind")
        if kind == "switch":
            slot["switches"] += 1
        elif kind == "probe":
            slot["probes"] += 1
        elif kind == "error":
            slot["errors"] += 1
        if row.get("latency_ms") is not None:
            slot["latency_sum"] += float(row["latency_ms"])
            slot["latency_n"] += 1
    models = []
    for m, s in by_model.items():
        avg = (s["latency_sum"] / s["latency_n"]) if s["latency_n"] else None
        models.append({
            "model": m,
            "events": s["events"],
            "switches": s["switches"],
            "probes": s["probes"],
            "errors": s["errors"],
            "avg_latency_ms": round(avg, 2) if avg is not None else None,
        })
    return {
        "config": cfg,
        "presets": [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.1-70b-instruct",
            "google/gemini-2.0-flash-exp:free",
            "mistralai/mistral-7b-instruct:free",
        ],
        "analytics": sorted(models, key=lambda x: x["events"], reverse=True),
    }


@router.post("/llm")
async def update_llm(
    body: LlmUpdateRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    LLM_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if LLM_RUNTIME.exists():
        try:
            current = json.loads(LLM_RUNTIME.read_text(encoding="utf-8"))
        except Exception:
            current = {}

    if body.model:
        current["model"] = body.model.strip()
        os.environ["OPENROUTER_MODEL"] = current["model"]
    if body.api_key is not None and body.api_key.strip():
        current["api_key"] = body.api_key.strip()
        os.environ["OPENROUTER_API_KEY"] = current["api_key"]
    current["provider"] = body.provider or current.get("provider") or "openrouter"
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    LLM_RUNTIME.write_text(json.dumps(current, indent=2), encoding="utf-8")

    # Soft-update .env model line (never echo key back)
    _patch_dotenv_model(current.get("model"), current.get("api_key") if body.api_key else None)

    _append_jsonl(LLM_ANALYTICS, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "switch",
        "model": current.get("model"),
        "provider": current.get("provider"),
        "key_updated": bool(body.api_key and body.api_key.strip()),
    })
    return {
        "ok": True,
        "config": {
            "provider": current.get("provider"),
            "model": current.get("model"),
            "api_key_set": bool(current.get("api_key") or os.getenv("OPENROUTER_API_KEY")),
            "api_key_masked": _mask_key(current.get("api_key") or os.getenv("OPENROUTER_API_KEY") or ""),
        },
    }


def _patch_dotenv_model(model: Optional[str], api_key: Optional[str]) -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except Exception:
        return
    lines = text.splitlines()
    out = []
    seen_model = False
    seen_key = False
    for line in lines:
        if model and line.startswith("OPENROUTER_MODEL="):
            out.append(f"OPENROUTER_MODEL={model}")
            seen_model = True
        elif api_key and line.startswith("OPENROUTER_API_KEY="):
            out.append(f"OPENROUTER_API_KEY={api_key}")
            seen_key = True
        else:
            out.append(line)
    if model and not seen_model:
        out.append(f"OPENROUTER_MODEL={model}")
    if api_key and not seen_key:
        out.append(f"OPENROUTER_API_KEY={api_key}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
