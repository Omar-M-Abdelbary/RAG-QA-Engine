import time
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.schemas.question_schemas import QuestionRequest, QuestionResponse, HealthResponse, StatsResponse
from app.services.rag_service import RAGService
from app.infrastructure.database.crud import QueryCRUD
from app.utils.logger import get_logger

logger = get_logger(__name__)

class RAGController:
    """Controller for RAG operations"""
    
    def __init__(self):
        self.rag_service = None
    
    def _get_rag_service(self):
        """Lazy load RAG service (heavy operation)"""
        if self.rag_service is None:
            logger.info("Initializing RAG Service...")
            try:
                self.rag_service = RAGService()
            except Exception as e:
                logger.error(f"Failed to initialize RAG service: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="RAG service unavailable. Please try again later."
                )
        return self.rag_service
    
    async def ask_question(
        self, 
        request: QuestionRequest,
        db: Session
    ) -> QuestionResponse:
        """
        Handle question answering request
        
        Args:
            request: Question request
            db: Database session
            
        Returns:
            QuestionResponse
        """
        start_time = time.time()
        
        try:
            logger.info(f"Received question: '{request.question}'")
            
            # Get RAG service
            rag_service = self._get_rag_service()
            
            # Process question
            result = await rag_service.answer_question(
                question=request.question,
                top_k=request.top_k,
                use_cache=request.use_cache,
                return_metadata=True
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Get top similarity score
            similarity_score = None
            if 'metadata' in result and 'contexts' in result['metadata']:
                contexts = result['metadata']['contexts']
                if contexts:
                    similarity_score = contexts[0].get('score')
            
            # Get quality score from validation
            quality_score = 0.0
            if 'metadata' in result and 'validation' in result['metadata']:
                quality_score = result['metadata']['validation'].get('quality_score', 0.0)
            
            # Save to database (don't fail if this fails)
            try:
                query_data = {
                    'question': request.question,
                    'answer': result['answer'],
                    'success': result['success'],
                    'response_time': response_time,
                    'quality_score': quality_score,
                    'num_contexts': result.get('num_contexts', 0),
                    'cached': result.get('cached', False),
                    'similarity_score': similarity_score
                }
                QueryCRUD.create_query(db, query_data)
                logger.info("Query saved to database")
            except Exception as db_error:
                logger.error(f"Failed to save query to database: {db_error}")
                # Continue - database failure shouldn't break user experience
            
            # Return response
            return QuestionResponse(
                question=request.question,
                answer=result['answer'],
                success=result['success'],
                response_time=round(response_time, 3),
                cached=result.get('cached', False),
                similarity_score=round(similarity_score, 4) if similarity_score else None,
                num_contexts=result.get('num_contexts', 0)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your question."
            )
    
    async def health_check(self) -> HealthResponse:
        """
        Health check endpoint
        
        Returns:
            HealthResponse
        """
        try:
            components = {
                "database": "unknown",
                "rag_service": "not_initialized"
            }
            
            # Check if RAG service is initialized
            if self.rag_service is not None:
                components["rag_service"] = "healthy"
                
                # Check FAISS
                if hasattr(self.rag_service.retrieval_service, 'index'):
                    index = self.rag_service.retrieval_service.index
                    components["faiss_index"] = f"healthy ({index.ntotal:,} vectors)"
                
                # Check LLM
                if self.rag_service.llm_client:
                    components["llm_client"] = "healthy"
            
            return HealthResponse(
                status="healthy",
                message="RAG system is operational",
                timestamp=datetime.utcnow(),
                components=components
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                message=f"System check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                components={}
            )
    
    async def get_stats(self, db: Session) -> StatsResponse:
        """
        Get system statistics
        
        Args:
            db: Database session
            
        Returns:
            StatsResponse
        """
        try:
            stats = QueryCRUD.get_query_stats(db)
            frequent_questions = QueryCRUD.get_most_frequent_questions(db, limit=10)
            
            return StatsResponse(
                total_queries=stats['total_queries'],
                successful_queries=stats['successful_queries'],
                success_rate=stats['success_rate'],
                avg_response_time=stats['avg_response_time'],
                avg_quality_score=stats['avg_quality_score'],
                cache_hit_rate=stats['cache_hit_rate'],
                most_frequent_questions=frequent_questions
            )
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error retrieving statistics"
            )