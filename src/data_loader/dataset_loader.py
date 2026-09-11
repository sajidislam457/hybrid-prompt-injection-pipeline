"""
Data loader for prompt injection datasets
"""

import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptSample:
    """Unified prompt sample format."""
    text: str
    label: int  # 0=benign, 1=malicious
    attack_category: Optional[str] = None
    severity: Optional[str] = None
    group_id: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class DatasetLoader:
    """Loads and processes prompt injection datasets."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples: List[PromptSample] = []
    
    def explore_files(self):
        """Explore data files in raw directory."""
        files = list(self.raw_dir.glob("*"))
        if not files:
            logger.warning("No files found in data/raw/")
            return
        
        logger.info("\nData files found:")
        for file_path in files:
            if file_path.name.startswith('.'):
                continue
            logger.info(f"  - {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{'):
                        data = json.loads(first_line)
                        logger.info(f"     Keys: {list(data.keys())}")
                        text = data.get('text', data.get('payload', 'N/A'))
                        logger.info(f"     Text preview: {str(text)[:80]}...")
            except Exception as e:
                logger.warning(f"     Error reading: {e}")
    
    def load_jayavibhav(self) -> List[PromptSample]:
        """Load jayavibhav prompt injection dataset."""
        samples = []
        file_path = self.raw_dir / "jayavibhav_prompt_injection.jsonl"
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return samples
        
        logger.info(f"Loading {file_path.name}...")
        
        # Use utf-8-sig to handle BOM if present
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            total = len(lines)
            logger.info(f"  Total lines: {total}")
            
            for i, line in enumerate(lines):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    text = data.get('text', '')
                    if not text:
                        continue
                    
                    label = data.get('label', 0)
                    if isinstance(label, str):
                        label = 1 if label.lower() in ['malicious', '1', 'true'] else 0
                    elif isinstance(label, bool):
                        label = 1 if label else 0
                    else:
                        label = int(label) if label is not None else 0
                    
                    attack_category = self._classify_attack(text) if label == 1 else None
                    
                    samples.append(PromptSample(
                        text=text,
                        label=label,
                        attack_category=attack_category,
                        source="jayavibhav",
                        group_id=f"jay_{hashlib.md5(text.encode('utf-8')).hexdigest()[:10]}"
                    ))
                except (json.JSONDecodeError, Exception):
                    continue
                
                if (i + 1) % 50000 == 0:
                    logger.info(f"  Processed {i+1} lines...")
        
        logger.info(f"Loaded {len(samples)} samples from jayavibhav")
        return samples
    
    def load_moltbook(self) -> List[PromptSample]:
        """Load moltbook extended dataset."""
        samples = []
        file_path = self.raw_dir / "moltbook_extended.jsonl"
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return samples
        
        logger.info(f"Loading {file_path.name}...")
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            total = len(lines)
            logger.info(f"  Total lines: {total}")
            
            for i, line in enumerate(lines):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    
                    text = data.get('payload', '')
                    if not text:
                        text = data.get('text', data.get('prompt', ''))
                    if not text:
                        continue
                    
                    label = 0
                    categories = str(data.get('categories', '')).lower()
                    keywords = str(data.get('keywords', '')).lower()
                    wrapper = str(data.get('wrapper', '')).lower()
                    
                    injection_indicators = [
                        'injection', 'jailbreak', 'attack', 'malicious',
                        'override', 'ignore', 'bypass', 'disregard', 'forget'
                    ]
                    
                    combined = f"{categories} {keywords} {wrapper}"
                    if any(ind in combined for ind in injection_indicators):
                        label = 1
                    
                    text_lower = text.lower()
                    if any(ind in text_lower for ind in ['ignore previous', 'forget all', 'bypass safety']):
                        label = 1
                    
                    attack_category = self._classify_attack(text) if label == 1 else None
                    
                    samples.append(PromptSample(
                        text=text,
                        label=label,
                        attack_category=attack_category,
                        source="moltbook",
                        group_id=f"molt_{data.get('id', hashlib.md5(text.encode('utf-8')).hexdigest()[:10])}"
                    ))
                except (json.JSONDecodeError, Exception):
                    continue
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"  Processed {i+1} lines...")
        
        logger.info(f"Loaded {len(samples)} samples from moltbook")
        return samples
    
    def _classify_attack(self, text: str) -> str:
        """Heuristically classify attack type."""
        text_lower = text.lower()
        
        patterns = {
            "direct_injection": ['ignore', 'forget', 'override', 'disregard', 'bypass'],
            "jailbreak": ['dan', 'jailbreak', 'developer mode', 'do anything', 'unrestricted'],
            "system_extraction": ['system prompt', 'configuration', 'internal rules', 'safety guidelines'],
            "data_extraction": ['extract', 'reveal', 'expose', 'list all', 'output all'],
            "tool_injection": ['execute', 'function', 'api call', 'sql', 'run command'],
            "context_poisoning": ['remember', 'adopt', 'persona', 'pretend', 'role-play'],
            "obfuscation": ['base64', 'rot13', 'encoded', 'decode', 'hex'],
            "multi_turn": ['first', 'then', 'step', 'gradually', 'next']
        }
        
        for category, keywords in patterns.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return "unknown"
    
    def build_dataset(self) -> Tuple[List[PromptSample], Dict]:
        """Build unified dataset."""
        all_samples = []
        stats = {}
        
        jay_samples = self.load_jayavibhav()
        if jay_samples:
            all_samples.extend(jay_samples)
            stats['jayavibhav'] = len(jay_samples)
        
        molt_samples = self.load_moltbook()
        if molt_samples:
            all_samples.extend(molt_samples)
            stats['moltbook'] = len(molt_samples)
        
        if not all_samples:
            logger.error("No samples loaded!")
            return [], stats
        
        unique_samples = []
        seen_texts = set()
        for sample in all_samples:
            key = hashlib.md5(sample.text[:100].encode('utf-8')).hexdigest()
            if key not in seen_texts:
                seen_texts.add(key)
                unique_samples.append(sample)
        
        stats['total'] = len(unique_samples)
        stats['malicious'] = sum(1 for s in unique_samples if s.label == 1)
        stats['benign'] = sum(1 for s in unique_samples if s.label == 0)
        
        self.samples = unique_samples
        logger.info(f"\nBuilt dataset: {len(unique_samples)} samples")
        logger.info(f"   Malicious: {stats['malicious']}")
        logger.info(f"   Benign: {stats['benign']}")
        
        return unique_samples, stats

    @staticmethod
    def sample_group_id(sample: Any) -> str:
        """Stable group key: stored group_id, else a hash of the text."""
        if isinstance(sample, dict):
            gid = sample.get("group_id")
            text = sample.get("text") or ""
            label = sample.get("label")
        else:
            gid = getattr(sample, "group_id", None)
            text = getattr(sample, "text", "") or ""
            label = getattr(sample, "label", None)
        if gid:
            return str(gid)
        payload = f"{text}|{label}"
        return f"group_{hashlib.md5(payload.encode('utf-8')).hexdigest()[:10]}"

    @staticmethod
    def sample_label(sample: Any) -> int:
        if isinstance(sample, dict):
            return int(sample.get("label") or 0)
        return int(getattr(sample, "label", 0) or 0)

    @staticmethod
    def sample_text(sample: Any) -> str:
        if isinstance(sample, dict):
            return str(sample.get("text") or "")
        return str(getattr(sample, "text", "") or "")

    @classmethod
    def stratified_group_kfold_indices(
        cls,
        samples: Sequence[Any],
        n_splits: int = 5,
        random_state: int = 42,
    ) -> List[Dict]:
        """
        5-fold StratifiedGroupKFold: keep class rates similar across folds
        and never put the same group_id in both train and validation.
        """
        n = len(samples)
        if n < n_splits:
            raise ValueError(f"Need at least {n_splits} samples for {n_splits}-fold CV, got {n}")

        y = [cls.sample_label(s) for s in samples]
        groups = [cls.sample_group_id(s) for s in samples]
        n_groups = len(set(groups))
        n_pos_groups = len({g for g, lab in zip(groups, y) if lab == 1})
        n_neg_groups = len({g for g, lab in zip(groups, y) if lab == 0})
        max_splits = min(n_splits, n_groups, max(n_pos_groups, 1), max(n_neg_groups, 1))
        if max_splits < 2:
            raise ValueError(
                "StratifiedGroupKFold needs at least 2 groups per class; "
                f"got pos_groups={n_pos_groups} neg_groups={n_neg_groups}"
            )
        if max_splits < n_splits:
            logger.warning(
                "Reducing n_splits from %s to %s (groups/class too few for 5 folds)",
                n_splits,
                max_splits,
            )
            n_splits = max_splits

        dummy_x = [[0] for _ in range(n)]
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds: List[Dict] = []
        for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(dummy_x, y, groups)):
            train_idx = [int(i) for i in train_idx]
            val_idx = [int(i) for i in val_idx]
            train_g = {groups[i] for i in train_idx}
            val_g = {groups[i] for i in val_idx}
            leaked = train_g & val_g
            if leaked:
                raise RuntimeError(
                    f"Fold {fold_i} leaked {len(leaked)} groups between train and validation"
                )
            y_tr = [y[i] for i in train_idx]
            y_va = [y[i] for i in val_idx]
            folds.append({
                "fold": fold_i,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_train_groups": len(train_g),
                "n_val_groups": len(val_g),
                "train_malicious": sum(y_tr),
                "train_benign": len(y_tr) - sum(y_tr),
                "val_malicious": sum(y_va),
                "val_benign": len(y_va) - sum(y_va),
                "train_pos_rate": (sum(y_tr) / len(y_tr)) if y_tr else 0.0,
                "val_pos_rate": (sum(y_va) / len(y_va)) if y_va else 0.0,
                "group_overlap": 0,
            })
        return folds

    def _write_jsonl(self, path: Path, split_samples: Sequence[PromptSample]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for sample in split_samples:
                payload = sample.to_dict() if hasattr(sample, "to_dict") else sample
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        logger.info("Saved %s samples to %s", len(split_samples), path.name)

    def save_splits(
        self,
        samples: Optional[List[PromptSample]] = None,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> Dict:
        """
        Partition with StratifiedGroupKFold (default 5 folds).

        Each sample appears in exactly one validation fold. Train / val / test
        jsonl files are 3 + 1 + 1 folds from that partition so groups never
        cross the production splits either.
        """
        if samples is None:
            samples = self.samples

        if not samples:
            logger.error("No samples to save!")
            return {}

        folds = self.stratified_group_kfold_indices(
            samples, n_splits=n_splits, random_state=random_state
        )
        n_splits = len(folds)

        all_path = self.processed_dir / "all.jsonl"
        self._write_jsonl(all_path, samples)

        folds_dir = self.processed_dir / "folds"
        folds_dir.mkdir(parents=True, exist_ok=True)
        for old in folds_dir.glob("fold_*_val.jsonl"):
            old.unlink()

        for fold in folds:
            val_samples = [samples[i] for i in fold["val_idx"]]
            self._write_jsonl(folds_dir / f"fold_{fold['fold']}_val.jsonl", val_samples)

        # Production files: first n-2 folds → train, penultimate → val, last → test.
        if n_splits >= 3:
            train_fold_ids = list(range(0, n_splits - 2))
            val_fold_id = n_splits - 2
            test_fold_id = n_splits - 1
        elif n_splits == 2:
            train_fold_ids = [0]
            val_fold_id = 1
            test_fold_id = 1
        else:
            train_fold_ids = [0]
            val_fold_id = 0
            test_fold_id = 0

        def _from_folds(fold_ids: List[int]) -> List[PromptSample]:
            out: List[PromptSample] = []
            for fid in fold_ids:
                out.extend(samples[i] for i in folds[fid]["val_idx"])
            return out

        splits = {
            "train": _from_folds(train_fold_ids),
            "val": _from_folds([val_fold_id]),
            "test": _from_folds([test_fold_id]),
        }

        for name, split_samples in splits.items():
            self._write_jsonl(self.processed_dir / f"{name}.jsonl", split_samples)

        def _pos_rate(rows: List[PromptSample]) -> float:
            if not rows:
                return 0.0
            return sum(1 for s in rows if s.label == 1) / len(rows)

        stats = {
            "splitter": "StratifiedGroupKFold",
            "n_splits": n_splits,
            "random_state": random_state,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": len(samples),
            "malicious": sum(1 for s in samples if s.label == 1),
            "benign": sum(1 for s in samples if s.label == 0),
            "train_pos_rate": _pos_rate(splits["train"]),
            "val_pos_rate": _pos_rate(splits["val"]),
            "test_pos_rate": _pos_rate(splits["test"]),
            "train_folds": train_fold_ids,
            "val_fold": val_fold_id,
            "test_fold": test_fold_id,
            "folds": [
                {k: v for k, v in fold.items() if k not in ("train_idx", "val_idx")}
                for fold in folds
            ],
        }

        manifest = {
            **stats,
            "all_path": str(all_path),
            "folds": folds,
        }
        with open(self.processed_dir / "cv_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with open(self.processed_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            "StratifiedGroupKFold n_splits=%s | train=%s val=%s test=%s | "
            "pos rates train=%.3f val=%.3f test=%.3f",
            n_splits,
            stats["train"],
            stats["val"],
            stats["test"],
            stats["train_pos_rate"],
            stats["val_pos_rate"],
            stats["test_pos_rate"],
        )
        return stats