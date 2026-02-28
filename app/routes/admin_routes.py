from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any

from app.controller.admin_controller import AdminController
from app.schemas.admin_schemas import (
    EmbeddingRequest,
    LLMTestRequest,
    PreprocessRequest
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize controller
admin_controller = AdminController()


@router.post("/embeddings/create", tags=["Embeddings"])
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for a list of texts
    
    Example:
```json
    {
        "texts": ["What is AI?", "Machine learning basics"],
        "normalize": true
    }
```
    """
    try:
        result = admin_controller.create_embeddings(
            texts=request.texts,
            normalize=request.normalize
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/embeddings/test", tags=["Embeddings"])
async def test_embedding_model():
    """Test if the embedding model is working"""
    try:
        result = admin_controller.test_embedding_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/test", tags=["LLM"])
async def test_llm_connection(request: LLMTestRequest):
    """
    Test LLM connection and generation
    
    Example:
```json
    {
        "prompt": "What is the capital of France?",
        "max_tokens": 50,
        "temperature": 0.7
    }
```
    """
    try:
        result = await admin_controller.test_llm_connection(
            prompt=request.prompt
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/llm/health", tags=["LLM"])
async def check_llm_health():
    """Check if LLM service is accessible"""
    result = await admin_controller.check_llm_health()
    return result


@router.post("/preprocess/text", tags=["Preprocessing"])
async def preprocess_text(request: PreprocessRequest):
    """
    Preprocess a single text
    
    Example:
```json
    {
        "text": "  This is SAMPLE text!  ",
        "options": {
            "lowercase": true,
            "remove_extra_spaces": true
        }
    }
```
    """
    try:
        result = admin_controller.preprocess_text(
            text=request.text,
            options=request.options
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preprocess/batch", tags=["Preprocessing"])
async def preprocess_batch(texts: List[str], options: Optional[Dict[str, Any]] = None):
    """Preprocess multiple texts at once"""
    try:
        result = admin_controller.preprocess_batch(texts=texts, options=options)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/system/info", tags=["System"])
async def get_system_info():
    """Get information about RAG system components"""
    try:
        result = admin_controller.get_system_info()
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}