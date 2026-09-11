"""5-fold StratifiedGroupKFold evaluation for Layer 2."""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.data_loader.dataset_loader import DatasetLoader
from src.layers.layer2_classifiers import Layer2Classifier
from src.training.team_weights import build_sample_weights

logger = logging.getLogger(__name__)


def _as_dict(sample: Any) -> Dict[str, Any]:
    if isinstance(sample, dict):
        return sample
    if hasattr(sample, "to_dict"):
        return sample.to_dict()
    return {
        "text": DatasetLoader.sample_text(sample),
        "label": DatasetLoader.sample_label(sample),
        "group_id": DatasetLoader.sample_group_id(sample),
        "source": getattr(sample, "source", None),
    }


def load_cv_universe(processed_dir: Path) -> List[Dict[str, Any]]:
    all_path = processed_dir / "all.jsonl"
    paths = [all_path] if all_path.exists() else [
        processed_dir / "train.jsonl",
        processed_dir / "val.jsonl",
        processed_dir / "test.jsonl",
    ]
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not (obj.get("text") or "").strip():
                    continue
                rows.append(obj)
    return rows


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "mean": float(sum(vals) / len(vals)),
        "std": float(std),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def run_layer2_stratified_group_cv(
    samples: Sequence[Any],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    max_features: int = 15000,
    ngram_range: tuple = (1, 2),
    team_weight: float = 50.0,
    team_sources: Optional[Sequence[str]] = None,
    folds: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Train Layer 2 on each training partition; score the held-out group fold."""
    rows = [_as_dict(s) for s in samples]
    if folds is None:
        folds = DatasetLoader.stratified_group_kfold_indices(
            rows, n_splits=n_splits, random_state=random_state
        )
    fold_metrics: List[Dict[str, Any]] = []
    for spec in folds:
        train_rows = [rows[i] for i in spec["train_idx"]]
        val_rows = [rows[i] for i in spec["val_idx"]]
        x_train = [r["text"] for r in train_rows]
        y_train = [int(r["label"]) for r in train_rows]
        x_val = [r["text"] for r in val_rows]
        y_val = [int(r["label"]) for r in val_rows]
        weights, wstats = build_sample_weights(
            train_rows, team_weight=team_weight, team_sources=team_sources
        )
        clf = Layer2Classifier(
            model_dir="./models/detector",
            max_features=max_features,
            ngram_range=ngram_range,
        )
        clf.train(x_train, y_train, sample_weight=weights, persist=False)
        metrics = clf.evaluate(x_val, y_val)
        metrics.update({
            "fold": spec.get("fold"),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_train_groups": spec.get("n_train_groups"),
            "n_val_groups": spec.get("n_val_groups"),
            "group_overlap": spec.get("group_overlap", 0),
            "train_pos_rate": spec.get("train_pos_rate"),
            "val_pos_rate": spec.get("val_pos_rate"),
            "team_rows": wstats.get("team_rows"),
        })
        fold_metrics.append(metrics)
        logger.info(
            "Fold %s val Acc=%.4f F1=%.4f AUC=%s n=%s groups_val=%s overlap=%s",
            spec.get("fold"),
            metrics["accuracy"],
            metrics["f1"],
            metrics.get("auc_roc"),
            metrics["n"],
            spec.get("n_val_groups"),
            spec.get("group_overlap", 0),
        )

    keys = ("accuracy", "f1", "f1_binary", "auc_roc")
    aggregate = {
        key: mean_std([m.get(key) for m in fold_metrics if m.get(key) is not None])
        for key in keys
    }
    return {
        "splitter": "StratifiedGroupKFold",
        "n_splits": len(folds),
        "random_state": random_state,
        "folds": fold_metrics,
        "aggregate": aggregate,
    }
