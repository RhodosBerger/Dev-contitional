"""
LLM API Routes 🧠
Exposes the OpenLLaMA / LlamaCppEngine via REST.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
# from backend.core.augmented.llm_processor import LLMProcessor

router = APIRouter(prefix="/llm", tags=["LLM"])

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    stop: Optional[List[str]] = None

@router.post("/generate")
async def generate_text(req: CompletionRequest):
    """
    Generates text using the active LLM engine.
    """
    # TODO: Get singleton LLMProcessor instance
    # result = processor.process(req.prompt)
    return {
        "text": f"Simulated completion for: {req.prompt[:20]}...",
        "model": "open_llama_3b_v2_q4",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20}
    }

@router.get("/models")
async def list_models():
    """Returns available models."""
    return {
        "models": [
            {"id": "open_llama_3b_v2", "backend": "llama_cpp"},
            {"id": "gpt-4", "backend": "openai"},
            {"id": "mock-model", "backend": "mock"}
        ]
    }

@router.get("/presets")
async def get_prompt_presets():
    """Returns a list of curated prompt presets."""
    pass # To be replaced with actual import
    try:
        from backend.core.prompt_library import PromptLibrary
        return {"presets": PromptLibrary.get_presets()}
    except ImportError:
        return {"presets": []}
