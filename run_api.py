#!/usr/bin/env python3
"""
Run the API server for Prompt Injection Defense System
"""

import os
import sys
import uvicorn
from pathlib import Path

# Windows consoles (cp1252) crash on emoji prints; force UTF-8 before any output.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("=" * 60)
    print("Prompt Injection Defense System API")
    print("=" * 60)
    print("Starting server...")
    print("API will be available at: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("=" * 60)
    print()
    print("Make sure you have trained the model first:")
    print("   python main.py --step train")
    print()
    print("=" * 60)

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False to prevent reload issues
        log_level="info",
        workers=1
    )