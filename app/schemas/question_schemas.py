from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class QuestionRequest(BaseModel):
    """Schema for question requests"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The question to ask the RAG system"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of context chunks to retrieve"
    )
    use_cache: Optional[bool] = Field(
        default=True,
        description="Whether to use cached responses"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the capital of France?",
                "top_k": 5,
                "use_cache": True
            }
        }

class QuestionResponse(BaseModel):
    """Schema for question responses"""
    question: str
    answer: str
    success: bool
    response_time: float
    cached: bool
    similarity_score: Optional[float] = None
    num_contexts: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the capital of France?",
                "answer": "The capital of France is Paris.",
                "success": True,
                "response_time": 2.3,
                "cached": False,
                "similarity_score": 0.85,
                "num_contexts": 3
            }
        }

class HealthResponse(BaseModel):
    """Schema for health check"""
    status: str
    message: str
    timestamp: datetime
    components: dict

class StatsResponse(BaseModel):
    """Schema for system statistics"""
    total_queries: int
    successful_queries: int
    success_rate: float
    avg_response_time: float
    avg_quality_score: float
    cache_hit_rate: float
    most_frequent_questions: List[dict]