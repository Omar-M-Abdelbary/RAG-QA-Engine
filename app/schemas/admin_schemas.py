from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    normalize: bool = Field(default=True)

class LLMTestRequest(BaseModel):
    prompt: str = Field(..., min_length=1)

class PreprocessRequest(BaseModel):
    text: str = Field(..., min_length=1)
    options: Optional[Dict[str, Any]] = Field(default={
        "lowercase": True,
        "remove_special_chars": False,
        "remove_numbers": False,
        "remove_extra_spaces": True
    })
