import time
from typing import Dict, Optional, List
from app.services.retrieval_service import RetrievalService
from app.services.query_processor import QueryProcessor
from app.services.response_validator import ResponseValidator
from app.clients.llm_client import LLMClient
from app.core.constants import RAGConstants
from app.core.config import Config
from app.utils.logger import get_logger
from app.utils.cache import RAGCache
from app.utils.performance_monitor import get_monitor

logger = get_logger(__name__)

class RAGService:
    """
    Enhanced RAG orchestrator with full Phase 2 features
    
    Features:
    - Query processing and enhancement
    - Smart retrieval
    - Response generation
    - Response validation
    - Caching
    - Performance monitoring
    """
    
    def __init__(self):
        logger.info("Initializing Enhanced RAG Service...")
        
        self.config = Config()
        
        # Initialize core components
        self.retrieval_service = RetrievalService()
        self.llm_client = LLMClient()
        
        # Initialize Phase 2 components
        self.query_processor = QueryProcessor()
        self.response_validator = ResponseValidator()
        self.cache = RAGCache(cache_dir="cache", ttl=3600)  # 1 hour cache
        self.monitor = get_monitor()
        
        # Conversation context (for multi-turn)
        self.conversation_history: List[Dict] = []
        
        logger.info("Enhanced RAG Service initialized successfully")
    
    async def answer_question(
        self, 
        question: str, 
        top_k: int = None,
        use_cache: bool = True,
        return_metadata: bool = False
    ) -> Dict:
        """
        Answer a question using full RAG pipeline with Phase 2 enhancements
        
        Args:
            question: User's question
            top_k: Number of contexts to retrieve
            use_cache: Whether to use caching
            return_metadata: Include detailed metadata in response
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        self.monitor.increment_counter('total_requests')
        
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        
        logger.info(f"Processing question: '{question}'")
        
        # STEP 1: Check cache
        if use_cache:
            cached_response = self.cache.get(question, top_k)
            if cached_response:
                self.monitor.increment_counter('cache_hits')
                logger.info("Returning cached response")
                
                total_time = time.time() - start_time
                self.monitor.record_latency('total', total_time)
                
                cached_response['cached'] = True
                return cached_response
        
        # STEP 2: Process query
        logger.debug("Processing query...")
        query_start = time.time()
        
        processed_query = self.query_processor.process_query(
            question, 
            context=self.conversation_history[-3:] if self.conversation_history else None
        )
        
        query_time = time.time() - query_start
        self.monitor.record_latency('query_processing', query_time)
        
        logger.info(f"Query processed - Keywords: {processed_query['keywords']}")
        
        # STEP 3: Retrieve contexts
        logger.debug("Starting retrieval phase...")
        retrieval_start = time.time()
        
        # Use cleaned query for retrieval
        retrieved_docs = self.retrieval_service.retrieve(
            processed_query['cleaned'], 
            top_k
        )
        
        retrieval_time = time.time() - retrieval_start
        self.monitor.record_latency('retrieval', retrieval_time)
        
        if not retrieved_docs:
            logger.warning("No documents retrieved for query")
            self.monitor.increment_counter('no_results')
            
            response = {
                'question': question,
                'answer': RAGConstants.NO_RESULTS_MESSAGE,
                'success': False,
                'cached': False
            }
            
            total_time = time.time() - start_time
            self.monitor.record_latency('total', total_time)
            
            return response
        
        logger.info(f"Retrieved {len(retrieved_docs)} contexts")
        
        # STEP 4: Extract contexts and scores
        contexts = [doc['chunk'] for doc in retrieved_docs]
        retrieval_scores = [doc['score'] for doc in retrieved_docs]
        
        # Format contexts for prompt
        context_text = "\n\n".join([
            f"Context {i+1}:\n{ctx}" 
            for i, ctx in enumerate(contexts)
        ])
        
        # STEP 5: Create prompt
        prompt = RAGConstants.RAG_PROMPT_TEMPLATE.format(
            context=context_text,
            question=question
        )
        
        # STEP 6: Generate answer
        logger.debug("Starting generation phase...")
        generation_start = time.time()
        
        self.monitor.increment_counter('llm_calls')
        
        answer = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=RAGConstants.SYSTEM_PROMPT
        )
        
        generation_time = time.time() - generation_start
        self.monitor.record_latency('generation', generation_time)
        
        logger.info("Answer generated successfully")
        
        # STEP 7: Validate response
        logger.debug("Validating response...")
        validation_start = time.time()
        
        validation_result = self.response_validator.validate_response(
            question=question,
            answer=answer,
            contexts=contexts,
            retrieval_scores=retrieval_scores
        )
        
        validation_time = time.time() - validation_start
        self.monitor.record_latency('validation', validation_time)
        
        logger.info(f"Response quality score: {validation_result['quality_score']:.2f}")
        
        # Check if response should be rejected
        if not validation_result['is_valid']:
            logger.warning(f"Response validation failed - recommendation: {validation_result['recommendation']}")
            self.monitor.increment_counter('invalid_responses')
            
            # Provide fallback answer
            answer = RAGConstants.INSUFFICIENT_INFO_MESSAGE
        
        # STEP 8: Build response
        response = {
            'question': question,
            'answer': answer,
            'success': validation_result['is_valid'],
            'num_contexts': len(retrieved_docs),
            'cached': False
        }
        
        # Add metadata if requested
        if return_metadata:
            response['metadata'] = {
                'processed_query': processed_query,
                'validation': validation_result,
                'contexts': [
                    {
                        'text': doc['chunk'][:200] + '...',
                        'score': doc['score']
                    }
                    for doc in retrieved_docs
                ],
                'performance': {
                    'query_processing_time': query_time,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    'validation_time': validation_time
                }
            }
        
        # STEP 9: Cache response (if valid)
        if validation_result['is_valid'] and use_cache:
            self.cache.set(question, top_k, response)
        
        # STEP 10: Update conversation history
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'timestamp': time.time()
        })
        
        # Keep only last 10 exchanges
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Record total time
        total_time = time.time() - start_time
        self.monitor.record_latency('total', total_time)
        
        logger.info(f"Request completed in {total_time:.3f}s")
        
        return response
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.monitor.get_stats()
        cache_stats = self.cache.get_stats()
        
        return {
            'performance': stats,
            'cache': cache_stats
        }
    
    def clear_cache(self):
        """Clear response cache"""
        logger.info("Clearing cache...")
        self.cache.clear()
    
    def reset_conversation(self):
        """Reset conversation history"""
        logger.info("Resetting conversation history")
        self.conversation_history = []