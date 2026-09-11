#!/usr/bin/env python3
"""
Fill the System Performance Comparison table (spc.pdf).

Scores each HuggingFace prompt-injection baseline as a *standalone* classifier
on the same held-out set as Check_Accuracy --mode heldout / ablation (test.jsonl).
Does not train those models. Does not overwrite models/detector/*.pkl.

Usage (from the Prompt project folder):
  python scripts/compare_baselines.py
  python scripts/compare_baselines.py --limit 2000
  python scripts/compare_baselines.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.hw_plan import apply_eval_runtime  # noqa: E402

apply_eval_runtime()


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    tp = tn = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        else:
            fn += 1
    n = max(len(y_true), 1)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "n": len(y_true),
        "accuracy": round(safe_div(tp + tn, n), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(safe_div(fp, fp + tn), 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def compute_auc_roc(y_true: List[int], y_score: List[float]) -> Optional[float]:
    if len(y_true) < 2 or len(set(int(y) for y in y_true)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return round(float(roc_auc_score(y_true, y_score)), 4)
    except Exception:
        return None


def load_heldout_rows(path: Path, limit: int, seed: int) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Held-out file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            rows.append({"text": text, "label": int(obj.get("label") or 0)})
    if limit and limit > 0:
        import random
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:limit]
    return rows

BASELINES = [
    {
        "key": "protectai_v2",
        "name": "ProtectAI DeBERTa-v3 prompt-injection v2",
        "hf_id": "protectai/deberta-v3-base-prompt-injection-v2",
    },
    {
        "key": "protectai_v1",
        "name": "ProtectAI DeBERTa-v3 prompt-injection v1",
        "hf_id": "protectai/deberta-v3-base-prompt-injection",
    },
    {
        "key": "fmops_distilbert",
        "name": "Fmops DistilBERT prompt-injection",
        "hf_id": "fmops/distilbert-prompt-injection",
    },
]

MALICIOUS_TOKENS = ("inject", "malicious", "unsafe", "attack", "label_1")
BENIGN_TOKENS = ("safe", "benign", "legitimate", "label_0")


def decode_hf(raw: Dict[str, Any]) -> Tuple[int, float]:
    """Return (pred_label 0/1, injection_probability) from a text-classification row."""
    label = str(raw.get("label", "")).lower()
    score = float(raw.get("score", 0.0))
    is_mal = any(t in label for t in MALICIOUS_TOKENS)
    if any(t in label for t in BENIGN_TOKENS):
        is_mal = False
    if "inject" in label or "unsafe" in label or "malicious" in label:
        is_mal = True
    risk = score if is_mal else 1.0 - score
    pred = 1 if risk >= 0.5 else 0
    return pred, float(risk)


def score_model(
    hf_id: str,
    texts: List[str],
    *,
    device: int,
    batch_size: int,
) -> Tuple[List[int], List[float], float]:
    from transformers import pipeline
    import torch

    clf = pipeline(
        "text-classification",
        model=hf_id,
        truncation=True,
        max_length=512,
        device=device,
    )
    preds: List[int] = []
    scores: List[float] = []
    t0 = time.perf_counter()
    ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
    clipped = [(t or "")[:2000] for t in texts]
    with ctx:
        i = 0
        while i < len(clipped):
            chunk = clipped[i : i + batch_size]
            raws = clf(chunk, batch_size=len(chunk), truncation=True, max_length=512)
            if isinstance(raws, dict):
                raws = [raws]
            for raw in raws:
                pred, risk = decode_hf(raw)
                preds.append(pred)
                scores.append(risk)
            i += len(chunk)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    mean_ms = elapsed_ms / max(len(texts), 1)
    del clf
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return preds, scores, mean_ms


def our_row_from_report(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    m = None
    if data.get("mode") == "ablation":
        m = ((data.get("ablations") or {}).get("full") or {}).get("metrics")
    elif data.get("metrics"):
        m = data["metrics"]
    if not m:
        return None
    return {
        "name": "Our full hybrid pipeline",
        "accuracy": m.get("accuracy"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "f1": m.get("f1"),
        "auc_roc": m.get("auc_roc"),
        "fpr": m.get("fpr"),
        "ms": m.get("latency_ms_mean"),
        "n": m.get("n"),
        "source": str(path),
    }


def fmt(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def main() -> int:
    parser = argparse.ArgumentParser(description="System performance comparison vs HF baselines")
    parser.add_argument("--test-path", type=str, default=str(ROOT / "data/processed/test.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="0 = full test.jsonl (same as paper held-out)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--our-report",
        type=str,
        default=str(ROOT / "logs/paper/check_accuracy_ablation/report.json"),
        help="Existing ablation/heldout JSON for the 'Our full hybrid pipeline' row",
    )
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "logs"))
    args = parser.parse_args()

    rows = load_heldout_rows(Path(args.test_path), limit=args.limit, seed=args.seed)
    if not rows:
        print(f"[ERROR] No labeled rows in {args.test_path}")
        print("Run: python main.py --step process")
        return 1

    y_true = [int(r["label"]) for r in rows]
    texts = [r["text"] for r in rows]
    print(f"Held-out n={len(rows)} pos={sum(y_true)} neg={len(y_true) - sum(y_true)}")
    print(f"File: {args.test_path}")

    device = -1
    if args.device in {"auto", "cuda"}:
        try:
            import torch
            if torch.cuda.is_available():
                device = 0
        except Exception:
            device = -1
    print(f"Device: {'cuda:0' if device >= 0 else 'cpu'} | batch={args.batch_size}")

    table: List[Dict[str, Any]] = []
    ours = our_row_from_report(Path(args.our_report) if args.our_report else None)
    if ours:
        table.append(ours)
        print(f"Loaded hybrid row from {ours['source']}")
    else:
        print("[warn] Could not load hybrid metrics; table will list baselines only.")
        print("       Re-run: python scripts/Check_Accuracy.py --mode ablation --limit 0 --no-balance")

    for spec in BASELINES:
        print(f"\n=== {spec['name']} ===")
        print(f"    {spec['hf_id']}")
        try:
            preds, scores, mean_ms = score_model(
                spec["hf_id"],
                texts,
                device=device,
                batch_size=max(1, args.batch_size),
            )
        except Exception as exc:
            print(f"    FAILED: {exc}")
            table.append({"name": spec["name"], "error": str(exc)})
            continue
        metrics = compute_metrics(y_true, preds)
        auc = compute_auc_roc(y_true, scores)
        metrics["auc_roc"] = auc
        table.append({
            "name": spec["name"],
            "hf_id": spec["hf_id"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc_roc": auc,
            "fpr": metrics["fpr"],
            "ms": round(mean_ms, 2),
            "n": metrics["n"],
            "confusion_matrix": metrics["confusion_matrix"],
        })
        print(
            f"    Acc={metrics['accuracy']:.4f} Prec={metrics['precision']:.4f} "
            f"Rec={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
            f"AUC={auc} FPR={metrics['fpr']:.4f} ms={mean_ms:.2f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "system_performance_comparison.json"
    md_path = out_dir / "system_performance_comparison.md"
    payload = {
        "title": "System Performance Comparison",
        "test_path": str(args.test_path),
        "n": len(rows),
        "threshold": 0.5,
        "protocol": "Standalone HF classifiers on the same held-out test.jsonl as ablation/full.",
        "rows": table,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# System Performance Comparison",
        "",
        f"- Test set: `{args.test_path}` (n={len(rows)})",
        "- Baselines: standalone HuggingFace text-classification, threshold 0.5",
        "- Hybrid row: copied from existing ablation/heldout report when available",
        "",
        "| System | Acc | Prec | Rec | F1 | AUC | FPR | ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        if row.get("error"):
            lines.append(f"| {row['name']} | FAILED: {row['error']} | | | | | | |")
            continue
        ms = row.get("ms")
        ms_s = f"{float(ms):.2f}" if ms is not None else "—"
        lines.append(
            f"| {row['name']} | {fmt(row.get('accuracy'))} | {fmt(row.get('precision'))} | "
            f"{fmt(row.get('recall'))} | {fmt(row.get('f1'))} | {fmt(row.get('auc_roc'))} | "
            f"{fmt(row.get('fpr'))} | {ms_s} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
