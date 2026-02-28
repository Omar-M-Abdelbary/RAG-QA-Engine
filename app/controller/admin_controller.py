from typing import List, Dict, Any
import numpy as np
from app.services.embedding_service import EmbeddingService
from app.services.preprocessing_service import PreprocessingService
from app.clients.llm_client import LLMClient
from app.utils.logger import get_logger
from app.core.config import Config
import time

logger = get_logger(__name__)

class AdminController:
    """Controller for admin operations"""
    
    def __init__(self):
        self.config = Config()
        self.preprocessing_service = PreprocessingService()
        self.embedding_service = EmbeddingService() 
        self.llm_client = LLMClient()
    
    
    def create_embeddings(self, texts: List[str], normalize: bool = True) -> Dict[str, Any]:
        """
        Create embeddings for a list of texts
        
        Args:
            texts: List of text strings
            normalize: Whether to normalize embeddings (handled by EmbeddingService)
        
        Returns:
            Dictionary with embeddings and metadata
        """
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts")
         
            embeddings_array = self.embedding_service.embed_texts(texts, show_progress=False)
            
            embeddings_list = embeddings_array.tolist()
            
            return {
                "embeddings": embeddings_list,
                "dimension": self.config.EMBEDDING_DIMENSION,
                "count": len(embeddings_list),
                "model": self.config.EMBEDDING_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def test_embedding_model(self) -> Dict[str, Any]:
        """Test if embedding model is working"""
        try:
            logger.info("Testing embedding model...")
            
            test_text = "This is a test sentence for embedding generation."
            embedding_array = self.embedding_service.embed_single(test_text)
            embedding = embedding_array.tolist()
            
            return {
                "status": "success",
                "message": "Embedding model is working correctly",
                "model": self.config.EMBEDDING_MODEL,
                "embedding_dimension": len(embedding),
                "sample_text": test_text,
                "sample_embedding_preview": embedding[:5],
                "sample_embedding_stats": {
                    "min": float(np.min(embedding_array)),
                    "max": float(np.max(embedding_array)),
                    "mean": float(np.mean(embedding_array))
                }
            }
            
        except Exception as e:
            logger.error(f"Embedding test failed: {e}")
            raise
    
    async def test_llm_connection(
        self, 
        prompt: str
    ) -> Dict[str, Any]:
        """Test LLM connection and generation"""
        try:
            logger.info(f"Testing LLM with prompt: '{prompt[:50]}...'")
            
            start_time = time.time()
            

            response = await self.llm_client.generate(
                prompt=prompt
            )
            
            latency = (time.time() - start_time) * 1000
            
            logger.info(f"LLM test successful. Latency: {latency:.2f}ms")
            
            return {
                "status": "success",
                "prompt": prompt,
                "response": response,
                "latency_ms": round(latency, 2)
            }
            
        except Exception as e:
            logger.error(f"LLM test failed: {e}")
            raise
    
    async def check_llm_health(self) -> Dict[str, Any]:
        """Check if LLM service is accessible"""
        try:
            
            test_response = await self.llm_client.generate(
                prompt="Say 'OK' if you can read this.",
            )
            
            return {
                "status": "healthy",
                "message": "LLM service is accessible",
                "test_response": test_response
            }
            
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"LLM service is not accessible: {str(e)}",
                "model": None
            }
    
    def preprocess_text(
        self, 
        text: str, 
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Preprocess a single text"""
        try:
            logger.info("Preprocessing text...")
            
            processed_text, operations = self.preprocessing_service.preprocess(
                text=text,
                options=options or {}
            )
            
            return {
                "original_text": text,
                "preprocessed_text": processed_text,
                "operations_applied": operations,
                "original_length": len(text),
                "processed_length": len(processed_text)
            }
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def preprocess_batch(
        self, 
        texts: List[str], 
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Preprocess multiple texts"""
        try:
            logger.info(f"Preprocessing {len(texts)} texts in batch...")
            
            results = []
            for text in texts:
                processed_text, operations = self.preprocessing_service.preprocess(
                    text=text,
                    options=options or {}
                )
                results.append({
                    "original": text,
                    "processed": processed_text,
                    "operations": operations
                })
            
            return {
                "status": "success",
                "count": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Batch preprocessing failed: {e}")
            raise

    def get_system_info(self) -> Dict[str, Any]:
        """Get system component information"""
        try:
            # Lazy load services (don't load if not needed to avoid overhead)
            embedding_status = "not_loaded"
            llm_status = "not_loaded"
            
            if self.embedding_service is not None:
                embedding_status = "loaded"
            
            if self.llm_client is not None:
                llm_status = "loaded"
            
            return {
                "status": "operational",
                "components": {
                    "embedding_model": {
                        "name": self.config.EMBEDDING_MODEL,
                        "dimension": self.config.EMBEDDING_DIMENSION,
                        "status": embedding_status
                    },
                    "llm_model": {
                        "name": self.config.LLM_MODEL if hasattr(self.config, 'LLM_MODEL') else "groq-llm",
                        "provider": "Groq",
                        "status": llm_status
                    },
                    "vector_store": {
                        "type": "FAISS",
                        "path": str(self.config.FAISS_INDEX_PATH) if hasattr(self.config, 'FAISS_INDEX_PATH') else "Natural-Questions-Base/indexes/faiss_index.bin",
                        "status": "available"
                    }
                },
                "endpoints": {
                    "embeddings": ["/api/v1/admin/embeddings/create", "/api/v1/admin/embeddings/test"],
                    "llm": ["/api/v1/admin/llm/test", "/api/v1/admin/llm/health"],
                    "preprocessing": ["/api/v1/admin/preprocess/text", "/api/v1/admin/preprocess/batch"],
                    "system": ["/api/v1/admin/system/info"]
                }
            }
            
        except Exception as e:
            logger.error(f"System info retrieval failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }