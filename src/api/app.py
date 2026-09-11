from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import os
import uvicorn
import logging
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_env_path = Path(__file__).parent.parent.parent / ".env"


def _load_env_file(path: Path) -> None:
    """Load .env even if it was saved as UTF-16 (common on Windows editors)."""
    if not path.exists():
        return
    try:
        load_dotenv(path)
        return
    except UnicodeDecodeError:
        raw = path.read_bytes()
        if raw.startswith(b"\xff\xfe"):
            text = raw.decode("utf-16-le")
        elif raw.startswith(b"\xfe\xff"):
            text = raw.decode("utf-16-be")
        elif len(raw) > 1 and raw[1:2] == b"\x00":
            text = raw.decode("utf-16-le")
        else:
            text = raw.decode("utf-8", errors="replace")
        text = text.lstrip("\ufeff")
        path.write_text(text, encoding="utf-8", newline="\n")
        load_dotenv(path)


_load_env_file(_env_path)

from src.pipeline.pipeline import PromptInjectionPipeline
from src.api import admin_routes
from src.api.admin_routes import router as admin_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Prompt Injection Defense System", version="3.1.0")

# Public chat UI may call detect/health from localhost:3001.
# Admin routes are token-gated; CORS still limited to local origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = PromptInjectionPipeline(use_llm=False)
pipeline_loaded = False


def _admin_runtime():
    return pipeline, pipeline_loaded


# Avoid import-name shadowing bugs: admin routes read through this binder.
admin_routes.bind_runtime(_admin_runtime)
app.include_router(admin_router)


class ConversationalRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to analyze")
    conversation_id: Optional[str] = Field(None)
    user_message: Optional[str] = Field(None)
    safe_suggestion: Optional[str] = Field(None)


@app.on_event("startup")
async def startup_event():
    global pipeline_loaded
    logger.info("Starting API...")
    model_dir = Path("./models/detector")
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        pipeline_loaded = False
        return
    try:
        success = pipeline.load_models()
        pipeline_loaded = success
        if success:
            logger.info("Pipeline loaded successfully!")
        else:
            logger.error("Failed to load pipeline models")
    except Exception as e:
        logger.error(f"Error: {e}")
        pipeline_loaded = False


@app.post("/detect-conversational")
async def detect_conversational(request: ConversationalRequest):
    global pipeline_loaded

    logger.info("=" * 60)
    logger.info(f"REQUEST: {request.prompt[:60]}...")
    logger.info("=" * 60)

    if not pipeline_loaded:
        try:
            success = pipeline.load_models()
            if success:
                pipeline_loaded = True
            else:
                raise HTTPException(status_code=503, detail="Pipeline not loaded")
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

    try:
        result = pipeline.process_conversational(
            request.prompt,
            request.conversation_id,
            request.user_message,
            request.safe_suggestion,
        )

        # Shared Lab log: every blocked chat turn, no user identity
        if result.get("type") == "blocked" or result.get("is_malicious"):
            try:
                from src.utils.malicious_inbox import ingest
                ingest(
                    request.prompt,
                    attack_type=result.get("attack_type") or "unknown",
                    attack_display_name=result.get("attack_display_name") or "Unknown",
                    risk_score=float(result.get("risk_score") or 0),
                    action="BLOCK",
                    severity="high",
                    decision_source=result.get("decision_source") or "public_block",
                    source="public_block",
                )
            except Exception:
                logger.warning("lab inbox ingest from API skipped", exc_info=True)

        # Always keep a heuristic safe suggestion on blocked replies
        if result.get("type") == "blocked":
            suggestion = (result.get("suggestion") or "").strip() or "What would you like help with today?"
            result["suggestion"] = suggestion
            result["needs_clarification"] = False
            human = (result.get("response") or "").strip()
            if not human or human == suggestion:
                attack = result.get("attack_type") or "unknown"
                blurbs = {
                    "system_extraction": "It looks like it was asking for internal system details.",
                    "data_extraction": "It looks like it was trying to pull private or sensitive data.",
                    "tool_injection": "It looks like it was trying to run commands or tools directly.",
                    "jailbreak": "It looks like a jailbreak attempt to bypass safety rules.",
                    "story_jailbreak": "It looks like a story was used to try to bypass safety.",
                    "direct_override": "It looks like it was trying to override my instructions.",
                    "direct_injection": "It looks like conflicting instructions were injected into the prompt.",
                    "context_tampering": "It looks like it was trying to rewrite the conversation context.",
                    "multi_turn": "It looks like a step-by-step attempt to slip past safety checks.",
                    "obfuscation": "It looks like the meaning was hidden with encoding or odd wording.",
                    "emotional_manipulation": "It looks like emotional pressure was used to bypass safety.",
                    "role_impersonation": "It looks like it was trying to force a different role or persona.",
                    "indirect_injection": "It looks like external content was used to sneak in instructions.",
                    "unknown": "Our safety filter flagged this request.",
                }
                blurb = blurbs.get(attack, blurbs["unknown"])
                result["response"] = (
                    f"That request looks unsafe. {blurb} "
                    f'Safe alternative: "{suggestion}" '
                    "Reply yes to use this, or tell me what you meant."
                )
            if not result.get("attack_display_name"):
                from src.layers.attack_typer import AttackTypeDetector
                result["attack_display_name"] = AttackTypeDetector.display_name(
                    result.get("attack_type") or "unknown"
                )

        logger.info(
            "📤 RESPONSE: type=%s response=%s",
            result.get("type"),
            (result.get("response") or "")[:120],
        )
        return result

    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        fallback = "What would you like help with today?"
        return {
            "type": "blocked",
            "conversation_id": None,
            "response": (
                f'That request looks unsafe. Safe alternative: "{fallback}" '
                "Reply yes to use this, or tell me what you meant."
            ),
            "suggestion": fallback,
            "confirmed": False,
            "status": "waiting_for_response",
            "needs_clarification": False,
        }


@app.get("/health")
async def health_check():
    layer_cfg = pipeline.config.get("layers") or {}
    flags = pipeline.config.get("feature_flags") or {}
    return {
        "status": "healthy" if pipeline_loaded else "degraded",
        "pipeline_loaded": pipeline_loaded,
        "version": (pipeline.config.get("system") or {}).get("version", "4.0.0"),
        "feature_flags": flags,
        "layer5_intent_preserving": bool((layer_cfg.get("layer5") or {}).get("intent_preserving", True)),
        "layer2b_enabled": bool((layer_cfg.get("layer2b") or {}).get("enabled", True)),
        "layer4_enabled": bool((layer_cfg.get("layer4") or {}).get("enabled", True)),
        "retrieval_enabled": bool((layer_cfg.get("retrieval") or {}).get("enabled", True)),
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
