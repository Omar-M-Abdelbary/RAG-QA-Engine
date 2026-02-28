from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.controller.rag_controller import RAGController
from app.schemas.question_schemas import (
    QuestionRequest, 
    QuestionResponse,
    HealthResponse,
    StatsResponse
)
from app.infrastructure.database.session import get_db

router = APIRouter()
rag_controller = RAGController()

@router.post(
    "/ask-question",
    response_model=QuestionResponse,
    status_code=200,
    summary="Ask a Question",
    description="Submit a question and get an AI-generated answer based on the knowledge base",
    tags=["RAG"]
)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    """
    Ask a question to the RAG system
    
    - **question**: The question to ask (required, 1-500 characters)
    - **top_k**: Number of context chunks to retrieve (optional, 1-10, default: 5)
    - **use_cache**: Whether to use cached responses (optional, default: true)
    """
    return await rag_controller.ask_question(request, db)

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
    summary="Health Check",
    description="Check the health status of the RAG system",
    tags=["System"]
)
async def health_check():
    """
    Health check endpoint
    
    Returns the operational status of all system components
    """
    return await rag_controller.health_check()

@router.get(
    "/stats",
    response_model=StatsResponse,
    status_code=200,
    summary="Get Statistics",
    description="Get system statistics and analytics",
    tags=["Analytics"]
)
async def get_stats(db: Session = Depends(get_db)):
    """
    Get system statistics
    
    Returns analytics about queries, performance, and usage
    """
    return await rag_controller.get_stats(db)