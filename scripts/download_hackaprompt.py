"""Download gated HackAPrompt dataset into data/raw/hackaprompt/.

Requires: HF account + accepted terms on
https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset
and HF_TOKEN (or huggingface-cli login).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "hackaprompt"


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = None
    if not token:
        print(
            "No Hugging Face token found.\n"
            "1) Accept access: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset\n"
            "2) Set HF_TOKEN or run: huggingface-cli login\n"
            "3) Re-run this script.",
            file=sys.stderr,
        )
        return 1

    from huggingface_hub import snapshot_download

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hackaprompt/hackaprompt-dataset -> {OUT}", flush=True)
    path = snapshot_download(
        repo_id="hackaprompt/hackaprompt-dataset",
        repo_type="dataset",
        local_dir=str(OUT),
        token=token,
    )
    print(f"OK: {path}", flush=True)
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            print(f"  {p.relative_to(OUT)}  {p.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
