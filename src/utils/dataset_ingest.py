"""
Dataset ingest: normalize schema → map attack categories → dedupe → append to train.

Used by Admin Datasets so any future upload follows the same path.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml

from src.layers.attack_typer import DISPLAY_NAMES
from src.utils.malicious_inbox import fingerprint

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_PATH = ROOT / "configs" / "attack_category_map.yaml"
TRAIN_PATH = ROOT / "data" / "processed" / "train.jsonl"
VAL_PATH = ROOT / "data" / "processed" / "val.jsonl"
TEST_PATH = ROOT / "data" / "processed" / "test.jsonl"
INCOMING = ROOT / "data" / "incoming"

# Blocked from admin ingest: noisy one-class dumps. Original split files
# (jayavibhav / moltbook) may be re-uploaded; val/test fingerprints are skipped.
# Blocked from admin ingest: noisy one-class dumps. Original split files
# (jayavibhav / moltbook) may be re-uploaded; val/test fingerprints are skipped.
SKIP_TRAIN_SOURCES = frozenset({
    "hackaprompt_all",
    "tensor_trust_all",
})


def _source_is_blocked(source: str) -> bool:
    s = (source or "").strip().lower()
    if not s:
        return False
    if s in SKIP_TRAIN_SOURCES:
        return True
    return any(blocked in s for blocked in SKIP_TRAIN_SOURCES)

TEXT_KEYS = ("text", "prompt", "payload", "input", "query", "content", "message")
LABEL_KEYS = ("label", "is_malicious", "malicious", "y", "target")
CATEGORY_KEYS = (
    "attack_category",
    "attack_type",
    "category",
    "categories",
    "type",
    "attack",
    "label_name",
)


def _load_map(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or DEFAULT_MAP_PATH
    canonical = list(DISPLAY_NAMES.keys())
    aliases: Dict[str, str] = {}
    if p.exists():
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if isinstance(raw.get("canonical"), list) and raw["canonical"]:
                canonical = [str(x).strip() for x in raw["canonical"] if str(x).strip()]
            if isinstance(raw.get("aliases"), dict):
                aliases = {
                    _norm_key(k): str(v).strip()
                    for k, v in raw["aliases"].items()
                    if k is not None and v is not None
                }
        except Exception:
            logger.warning("Failed to load attack_category_map.yaml", exc_info=True)
    return {"canonical": canonical, "aliases": aliases, "path": str(p)}


def _norm_key(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def map_attack_category(
    raw: Any,
    *,
    mapping: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Map an external category string to a canonical type.
    Returns (canonical, attack_category_raw or None if empty).
    """
    mapping = mapping or _load_map()
    canonical_set = set(mapping["canonical"])
    aliases: Dict[str, str] = mapping["aliases"]

    if raw is None:
        return "unknown", None
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is None or str(raw).strip() == "":
        return "unknown", None

    raw_s = str(raw).strip()
    key = _norm_key(raw_s)
    if not key:
        return "unknown", raw_s

    # Benign marker with no attack type
    if key in {"benign", "safe", "clean", "normal"}:
        return "unknown", raw_s

    if key in canonical_set:
        return key, raw_s

    mapped = aliases.get(key)
    if mapped and mapped in canonical_set:
        return mapped, raw_s
    if mapped:
        # alias pointed at unknown / typo — still accept if in display names
        if mapped in DISPLAY_NAMES:
            return mapped, raw_s

    return "unknown", raw_s


def _pick(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    lower = {str(k).lower(): v for k, v in row.items()}
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
        if k.lower() in lower and lower[k.lower()] not in (None, ""):
            return lower[k.lower()]
    return None


def _parse_label(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "malicious", "injection", "attack", "unsafe", "positive"}:
        return 1
    if s in {"0", "false", "no", "benign", "safe", "clean", "negative"}:
        return 0
    try:
        return 1 if int(float(s)) != 0 else 0
    except Exception:
        return None


def normalize_row(
    row: Dict[str, Any],
    *,
    source: str,
    mapping: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Convert an arbitrary labeled row into train.jsonl schema."""
    mapping = mapping or _load_map()
    text = _pick(row, TEXT_KEYS)
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    label = _parse_label(_pick(row, LABEL_KEYS))
    # If category says benign and label missing → 0
    raw_cat = _pick(row, CATEGORY_KEYS)
    if label is None and raw_cat is not None and _norm_key(raw_cat) in {"benign", "safe", "clean"}:
        label = 0
    if label is None:
        # Default: presence of a malicious-looking category → 1, else skip
        if raw_cat is not None and _norm_key(raw_cat) not in {"benign", "safe", "clean", "unknown", ""}:
            label = 1
        else:
            return None

    cat, raw_s = map_attack_category(raw_cat, mapping=mapping)
    if int(label) == 0:
        # benign rows: keep unknown category
        cat = "unknown"
    elif cat == "unknown":
        guessed = _guess_attack_type(text)
        if guessed != "unknown":
            cat = guessed

    out: Dict[str, Any] = {
        "text": text,
        "label": int(label),
        "attack_category": cat,
        "source": (source or "upload").strip() or "upload",
    }
    if raw_s is not None and (cat == "unknown" or _norm_key(raw_s) != cat):
        out["attack_category_raw"] = raw_s
    return out


def iter_file_rows(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield dict rows from .jsonl / .json / .csv / .txt."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".jsonl", ".txt"}:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
            else:
                # plain text line → unlabeled skip (need label)
                yield {"text": line}
    elif suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        elif isinstance(data, dict):
            for key in ("data", "rows", "samples", "items"):
                if isinstance(data.get(key), list):
                    for obj in data[key]:
                        if isinstance(obj, dict):
                            yield obj
                    return
            yield data
    elif suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        for obj in reader:
            yield dict(obj)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def load_jsonl_fingerprints(path: Path) -> Set[str]:
    fps: Set[str] = set()
    if not path.exists():
        return fps
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp = fingerprint(str(row.get("text") or ""))
            if fp:
                fps.add(fp)
    return fps


def load_train_fingerprints(train_path: Path = TRAIN_PATH) -> Set[str]:
    return load_jsonl_fingerprints(train_path)


def load_heldout_fingerprints() -> Set[str]:
    return load_jsonl_fingerprints(VAL_PATH) | load_jsonl_fingerprints(TEST_PATH)


_TYPE_DETECTOR = None


def _guess_attack_type(text: str) -> str:
    global _TYPE_DETECTOR
    if _TYPE_DETECTOR is None:
        from src.layers.attack_typer import AttackTypeDetector
        _TYPE_DETECTOR = AttackTypeDetector()
    try:
        return str(_TYPE_DETECTOR.detect_type(text) or "unknown")
    except Exception:
        return "unknown"


def preview_ingest(
    path: Path,
    *,
    source: Optional[str] = None,
    limit_samples: int = 5,
) -> Dict[str, Any]:
    mapping = _load_map()
    src = (source or _source_from_filename(path.name)).strip()
    if _source_is_blocked(src):
        return {
            "ok": False,
            "path": str(path),
            "source": src,
            "error": f"source '{src}' is blocked from ingest (noisy dump)",
        }
    existing = load_train_fingerprints()
    heldout = load_heldout_fingerprints()
    seen_file: Set[str] = set()

    stats = {
        "rows_read": 0,
        "normalized": 0,
        "skipped_invalid": 0,
        "duplicates_in_file": 0,
        "duplicates_vs_train": 0,
        "duplicates_vs_heldout": 0,
        "would_append": 0,
        "label_counts": Counter(),
        "category_counts": Counter(),
        "raw_category_counts": Counter(),
        "unmapped_raw": Counter(),
    }
    samples: List[Dict[str, Any]] = []

    for row in iter_file_rows(path):
        stats["rows_read"] += 1
        raw_cat = _pick(row, CATEGORY_KEYS)
        if raw_cat is not None and str(raw_cat).strip():
            stats["raw_category_counts"][str(raw_cat).strip()] += 1

        norm = normalize_row(row, source=src, mapping=mapping)
        if not norm:
            stats["skipped_invalid"] += 1
            continue
        stats["normalized"] += 1
        stats["label_counts"][norm["label"]] += 1
        stats["category_counts"][norm["attack_category"]] += 1
        if norm.get("attack_category_raw") and norm["attack_category"] == "unknown":
            stats["unmapped_raw"][str(norm["attack_category_raw"])] += 1

        fp = fingerprint(norm["text"])
        if fp in seen_file:
            stats["duplicates_in_file"] += 1
            continue
        seen_file.add(fp)
        if fp in heldout:
            stats["duplicates_vs_heldout"] += 1
            continue
        if fp in existing:
            stats["duplicates_vs_train"] += 1
            continue
        stats["would_append"] += 1
        if len(samples) < limit_samples:
            samples.append(norm)

    return {
        "ok": True,
        "path": str(path),
        "source": src,
        "canonical": mapping["canonical"],
        "display_names": {k: DISPLAY_NAMES.get(k, k) for k in mapping["canonical"]},
        "stats": {
            **stats,
            "label_counts": dict(stats["label_counts"]),
            "category_counts": dict(stats["category_counts"]),
            "raw_category_counts": dict(stats["raw_category_counts"].most_common(40)),
            "unmapped_raw": dict(stats["unmapped_raw"].most_common(40)),
        },
        "samples": samples,
    }


def ingest_file_into_train(
    path: Path,
    *,
    source: Optional[str] = None,
    train_path: Path = TRAIN_PATH,
    dry_run: bool = False,
    progress_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Normalize + map + dedupe and append unique rows to train.jsonl only.
    Never writes val.jsonl / test.jsonl. Prompts already in val or test are skipped.
    """
    def _prog(stage: str, detail: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not progress_cb:
            return
        try:
            progress_cb(stage, detail, extra or {})
        except Exception:
            pass

    mapping = _load_map()
    src = (source or _source_from_filename(path.name)).strip()
    if _source_is_blocked(src):
        return {
            "ok": False,
            "path": str(path),
            "source": src,
            "error": f"source '{src}' is blocked from ingest (noisy dump)",
        }
    _prog("indexing", "Scanning train / val / test fingerprints…")
    existing = load_train_fingerprints(train_path)
    heldout = load_heldout_fingerprints()
    _prog(
        "indexing",
        f"Indexed {len(existing):,} training prompts",
    )
    seen_file: Set[str] = set()

    to_append: List[Dict[str, Any]] = []
    stats = Counter()
    cat_counts: Counter = Counter()
    unmapped: Counter = Counter()

    _prog("scanning", f"Reading {path.name}…")
    for row in iter_file_rows(path):
        stats["rows_read"] += 1
        if stats["rows_read"] % 2000 == 0:
            _prog("scanning", f"Read {stats['rows_read']:,} rows…")
        norm = normalize_row(row, source=src, mapping=mapping)
        if not norm:
            stats["skipped_invalid"] += 1
            continue
        stats["normalized"] += 1
        cat_counts[norm["attack_category"]] += 1
        if norm.get("attack_category_raw") and norm["attack_category"] == "unknown" and norm["label"] == 1:
            unmapped[str(norm["attack_category_raw"])] += 1

        fp = fingerprint(norm["text"])
        if not fp:
            stats["skipped_invalid"] += 1
            continue
        if fp in seen_file:
            stats["duplicates_in_file"] += 1
            continue
        seen_file.add(fp)
        if fp in heldout:
            stats["duplicates_vs_heldout"] += 1
            continue
        if fp in existing:
            stats["duplicates_vs_train"] += 1
            continue
        existing.add(fp)
        to_append.append(norm)
        stats["appended"] += 1

    if not dry_run and to_append:
        _prog("appending", f"Writing {len(to_append):,} new rows to train…")
        train_path.parent.mkdir(parents=True, exist_ok=True)
        with train_path.open("a", encoding="utf-8") as fh:
            for row in to_append:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    _prog("done", f"Added {len(to_append):,} rows")
    return {
        "ok": True,
        "dry_run": dry_run,
        "path": str(path),
        "source": src,
        "train_path": str(train_path),
        "stats": dict(stats),
        "category_counts": dict(cat_counts),
        "unmapped_raw": dict(unmapped.most_common(40)),
        "appended": len(to_append),
    }


def _source_from_filename(name: str) -> str:
    stem = Path(name).stem
    # strip timestamp prefix YYYYMMDD_HHMMSS_
    stem = re.sub(r"^\d{8}_\d{6}_", "", stem)
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return stem or "upload"


def list_incoming_files() -> List[Dict[str, Any]]:
    INCOMING.mkdir(parents=True, exist_ok=True)
    files = []
    for f in sorted(INCOMING.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.is_file() and f.suffix.lower() in {".jsonl", ".json", ".csv", ".txt"}:
            files.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "source_guess": _source_from_filename(f.name),
            })
    return files


def get_taxonomy() -> Dict[str, Any]:
    mapping = _load_map()
    return {
        "canonical": mapping["canonical"],
        "display_names": {k: DISPLAY_NAMES.get(k, k) for k in mapping["canonical"]},
        "alias_count": len(mapping["aliases"]),
        "map_path": mapping["path"],
    }
