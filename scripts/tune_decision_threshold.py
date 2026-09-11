#!/usr/bin/env python3
"""Tune Layer 3 decision_threshold on val.jsonl (never touch test)."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ACCURACY_DISABLE_DECISION_LOG", "1")


def load_rows(path: Path, limit: int, seed: int) -> list:
    pos, neg = [], []
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
            row = {"text": text, "label": int(obj.get("label") or 0)}
            (pos if row["label"] == 1 else neg).append(row)
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    if limit and limit > 0:
        half = max(1, limit // 2)
        rows = pos[:half] + neg[:half]
    else:
        rows = pos + neg
    rng.shuffle(rows)
    return rows


def metrics(y_true, y_pred):
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
    n = max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * prec * rec / max(prec + rec, 1e-9)) if (prec + rec) else 0.0
    acc = (tp + tn) / n
    fpr = fp / max(fp + tn, 1)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "fpr": round(fpr, 4),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--val-path", default=str(ROOT / "data/processed/val.jsonl"))
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--thresholds",
        default="0.48,0.50,0.52,0.54,0.56,0.58",
        help="Comma-separated Layer3 decision_threshold values",
    )
    args = ap.parse_args()

    from src.pipeline.pipeline import PromptInjectionPipeline

    rows = load_rows(Path(args.val_path), args.limit, args.seed)
    print(f"Val tune samples: {len(rows)} from {args.val_path}", flush=True)

    pipe = PromptInjectionPipeline()
    pipe.load_models()
    texts = [r["text"] for r in rows]
    y_true = [r["label"] for r in rows]

    # One forward pass caching Layer2 outputs via process_many internals is hard;
    # re-run process per threshold but reuse loaded models (GPU warm).
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    results = []
    for thr in thresholds:
        pipe.layer3.decision_threshold = thr
        print(f"\n=== threshold={thr:.2f} ===", flush=True)
        dets = pipe.process_many(texts) if hasattr(pipe, "process_many") else [pipe.process(t) for t in texts]
        y_pred = [1 if d.is_malicious else 0 for d in dets]
        m = metrics(y_true, y_pred)
        m["threshold"] = thr
        results.append(m)
        print(
            f"  F1={m['f1']:.4f} Acc={m['accuracy']:.4f} FPR={m['fpr']:.4f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f} FP={m['fp']} FN={m['fn']}",
            flush=True,
        )

    # Prefer F1, but require FPR not worse than ~best-era if possible.
    eligible = [r for r in results if r["fpr"] <= 0.10]
    pool = eligible if eligible else results
    best = max(pool, key=lambda r: (r["f1"], -r["fpr"]))
    print("\nBEST:", best, flush=True)

    out = ROOT / "logs" / "threshold_tune_val.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "best": best}, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
