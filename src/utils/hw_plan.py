"""
Pick a held-out / ablation runtime plan from this PC.

Eval is a separate process from Admin. We use the GPU for Layer 2B when there
is enough free VRAM, and we leave some CPU cores for the API so the UI does
not freeze. Thread env vars must be applied before NumPy / sklearn / torch load.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

# Split: GPU does Layer 2B only. CPU threads do sklearn Layer 2 / rest.
# Batch is fixed (default 32) — no drop ladder.
# Do NOT hard-cap CUDA with set_per_process_memory_fraction (breaks DeBERTa → heuristic).
# empty_cache only when dedicated used crosses the slowdown band (~3000 MB), not
# on a timer while VRAM is comfortably under 2500.
MAX_LAYER2B_BATCH = 32
VRAM_SLOWDOWN_MB = 3000
VRAM_LIGHT_MB = 2500  # below this: skip periodic empty_cache


def cap_layer2b_batch(n: int) -> int:
    try:
        return max(1, min(int(n), MAX_LAYER2B_BATCH))
    except (TypeError, ValueError):
        return 1


def cuda_mem_mb() -> Optional[tuple]:
    """(used_mb, total_mb, free_mb) for GPU 0, or None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info()
        free_mb = int(free_b // (1024 * 1024))
        total_mb = int(total_b // (1024 * 1024))
        used_mb = max(0, total_mb - free_mb)
        return used_mb, total_mb, free_mb
    except Exception:
        return None


def empty_cuda_cache() -> None:
    """Return unused activation blocks to the driver. Does not unload DeBERTa."""
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def is_cuda_oom(exc: BaseException) -> bool:
    s = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "out of memory" in s
        or "cuda oom" in s
        or "cudaerror" in name
        or name == "outofmemoryerror"
    )


def apply_gpu_vram_budget(plan: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """
    Soft VRAM policy for eval (no hard CUDA fraction).

    Past ~3000 MB dedicated this PC slows; we empty_cache there and keep
    batch fixed so DeBERTa stays on the transformer (no heuristic).
    """
    path = ""
    if plan:
        path = str(plan.get("path") or "")
    path = path or str(os.getenv("EVAL_HW_PATH") or "")
    device = str((plan or {}).get("layer2b_device") or os.getenv("LAYER2B_DEVICE") or "")
    if path != "gpu" and device.lower() not in {"cuda", "gpu", "cuda:0", "0"}:
        return None
    return (
        f"Layer2B: GPU transformer, fixed batch={MAX_LAYER2B_BATCH}; "
        f"empty_cache only if dedicated >= {VRAM_SLOWDOWN_MB} MB "
        f"(skip periodic clear while < {VRAM_LIGHT_MB} MB)"
    )


def _nvidia_gpus() -> List[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.free,memory.total,name",
        "--format=csv,noheader,nounits",
    ]
    kwargs: Dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": 8,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(cmd, **kwargs)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return []
    gpus: List[Dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            free_mb = int(float(parts[0]))
            total_mb = int(float(parts[1]))
        except ValueError:
            continue
        gpus.append({
            "free_mb": free_mb,
            "total_mb": total_mb,
            "name": ", ".join(parts[2:]).strip(),
        })
    return gpus


def _torch_cuda_ok() -> tuple[bool, str]:
    """True only if THIS Python build of torch can run CUDA (not just nvidia-smi)."""
    try:
        import torch
    except Exception as exc:
        return False, f"torch import failed ({exc})"
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, f"torch {torch.__version__} CUDA ok ({name})"
        ver = getattr(torch, "__version__", "?")
        return False, f"torch {ver} has no CUDA (CPU build or missing GPU runtime)"
    except Exception as exc:
        return False, f"torch CUDA probe failed ({exc})"


def _load_eval_config() -> Dict[str, Any]:
    try:
        from src.utils.helpers import load_config
        cfg = load_config(CONFIG_PATH) or {}
    except Exception:
        cfg = {}
    return cfg if isinstance(cfg, dict) else {}


def build_eval_plan(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(config or _load_eval_config())
    eval_cfg = cfg.get("eval") if isinstance(cfg.get("eval"), dict) else {}
    layer2b_cfg = ((cfg.get("layers") or {}) if isinstance(cfg.get("layers"), dict) else {}).get("layer2b") or {}
    if not isinstance(layer2b_cfg, dict):
        layer2b_cfg = {}

    logical = int(os.cpu_count() or 4)
    # 0 = auto: use every leftover thread after a small Admin reserve.
    reserve_raw = int(eval_cfg.get("reserve_cpu_cores", 0) or 0)
    if reserve_raw <= 0:
        reserve = 2 if logical >= 8 else 1
    else:
        reserve = max(1, min(reserve_raw, max(1, logical - 1)))
    min_free = max(256, int(eval_cfg.get("min_gpu_free_mb", 1800)))
    cpu_cap = int(eval_cfg.get("cpu_thread_cap", 0) or 0)
    gpu_cpu = int(eval_cfg.get("gpu_cpu_threads", 0) or 0)
    leftover = max(1, logical - reserve)

    def _threads(cap: int) -> int:
        if cap <= 0:
            return leftover
        return max(1, min(cap, leftover))

    auto_hw = bool(eval_cfg.get("auto_hw", True))
    gpus = _nvidia_gpus()
    best = max(gpus, key=lambda g: g.get("free_mb", 0)) if gpus else None
    torch_ok, torch_note = _torch_cuda_ok()

    plan: Dict[str, Any] = {
        "auto_hw": auto_hw,
        "cpu_logical": logical,
        "reserve_cpu_cores": reserve,
        "min_gpu_free_mb": min_free,
        "gpus": gpus,
        "path": "cpu",
        "layer2b_device": "cpu",
        "omp_threads": 1,
        "reason": "",
        "gpu_name": None,
        "gpu_free_mb": None,
        "torch_cuda": torch_ok,
        "torch_note": torch_note,
        "layer2b_batch_size": 1,
    }

    if not auto_hw:
        plan["reason"] = "eval.auto_hw is false — single CPU thread (legacy)"
        plan["layer2b_device"] = str(layer2b_cfg.get("device") or "cpu")
        if str(plan["layer2b_device"]).lower() in {"cuda", "gpu", "cuda:0"}:
            plan["path"] = "gpu"
            plan["omp_threads"] = _threads(gpu_cpu)
        return plan

    vram_ok = bool(best and int(best.get("free_mb") or 0) >= min_free)
    gpu_ok = bool(vram_ok and torch_ok)
    if gpu_ok:
        plan["path"] = "gpu"
        plan["layer2b_device"] = "cuda"
        plan["omp_threads"] = _threads(gpu_cpu)
        plan["gpu_name"] = best.get("name")
        plan["gpu_free_mb"] = best.get("free_mb")
        cfg_b = int(eval_cfg.get("layer2b_batch_size") or MAX_LAYER2B_BATCH)
        plan["layer2b_batch_size"] = cap_layer2b_batch(cfg_b)
        plan["reason"] = (
            f"GPU {best.get('name')} Layer2B (transformer); "
            f"{best.get('free_mb')} MB free at start (need >= {min_free}); {torch_note}; "
            f"CPU sklearn {plan['omp_threads']} threads, "
            f"{reserve} cores reserved for Admin; Layer2B batch={plan['layer2b_batch_size']} fixed"
        )
        return plan

    omp = _threads(cpu_cap)
    plan["path"] = "cpu"
    plan["layer2b_device"] = "cpu"
    plan["omp_threads"] = omp
    if best:
        plan["gpu_name"] = best.get("name")
        plan["gpu_free_mb"] = best.get("free_mb")
        if not torch_ok:
            plan["reason"] = (
                f"GPU {best.get('name')} is present but {torch_note}; "
                f"CPU path with {omp} threads, {reserve} cores reserved for Admin"
            )
        else:
            plan["reason"] = (
                f"GPU {best.get('name')} only has {best.get('free_mb')} MB free "
                f"(need >= {min_free}); CPU path with {omp} threads, "
                f"{reserve} cores reserved for Admin"
            )
    else:
        plan["reason"] = (
            f"No NVIDIA GPU detected; CPU path with {omp} threads, "
            f"{reserve} cores reserved for Admin"
        )
    return plan


def apply_plan_to_environ(plan: Mapping[str, Any], environ: MutableMapping[str, str]) -> None:
    threads = str(int(plan.get("omp_threads") or 1))
    environ["OMP_NUM_THREADS"] = threads
    environ["MKL_NUM_THREADS"] = threads
    environ["OPENBLAS_NUM_THREADS"] = threads
    environ["NUMEXPR_NUM_THREADS"] = threads
    environ["LAYER2B_DEVICE"] = str(plan.get("layer2b_device") or "cpu")
    environ["EVAL_HW_APPLIED"] = "1"
    batch = cap_layer2b_batch(int(plan.get("layer2b_batch_size") or 1))
    if isinstance(plan, dict):
        plan["layer2b_batch_size"] = batch
    environ["EVAL_HW_JSON"] = json.dumps(dict(plan), ensure_ascii=True)
    environ["EVAL_HW_PATH"] = str(plan.get("path") or "cpu")
    environ["EVAL_L2B_BATCH"] = str(batch)
    # Less CUDA fragmentation on 4 GB cards (speed only; same model/gates).
    if not str(environ.get("PYTORCH_CUDA_ALLOC_CONF") or "").strip():
        environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def apply_eval_runtime(environ: Optional[MutableMapping[str, str]] = None) -> Dict[str, Any]:
    """
    Apply (or reuse) the eval hardware plan.

    Pass a copied env dict from Admin so the API process itself is unchanged.
    Call with no args at the top of Check_Accuracy before importing the pipeline.
    """
    target: MutableMapping[str, str] = environ if environ is not None else os.environ
    # Must be set before torch is imported (build_eval_plan probes CUDA).
    if not str(target.get("PYTORCH_CUDA_ALLOC_CONF") or "").strip():
        target["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if str(target.get("EVAL_HW_APPLIED") or "") == "1":
        raw = target.get("EVAL_HW_JSON") or ""
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed.get("path"):
                    if int(parsed.get("layer2b_batch_size") or 0) >= 1:
                        parsed["layer2b_batch_size"] = cap_layer2b_batch(
                            int(parsed.get("layer2b_batch_size") or 1)
                        )
                        apply_plan_to_environ(parsed, target)
                        return parsed
            except (json.JSONDecodeError, TypeError):
                pass
    plan = build_eval_plan()
    apply_plan_to_environ(plan, target)
    return plan


def describe_plan(plan: Mapping[str, Any]) -> str:
    path = str(plan.get("path") or "cpu")
    if path == "gpu":
        name = plan.get("gpu_name") or "CUDA"
        return f"GPU ({name})"
    return f"CPU ({plan.get('omp_threads', 1)} threads)"
