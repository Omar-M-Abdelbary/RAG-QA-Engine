import faiss
import json
from sentence_transformers import SentenceTransformer
from app.core.config import Config
from app.utils.logger import get_logger
from typing import List, Dict

logger = get_logger(__name__)

class RetrievalService:
    """Memory-efficient retrieval service"""
    
    def __init__(self):
        self.config = Config()
        
        logger.info("Loading FAISS index...")
        self.index = faiss.read_index(self.config.FAISS_INDEX_PATH)
        logger.info(f"Loaded {self.index.ntotal:,} vectors")
        
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
        
        self.metadata_path = self.config.FAISS_METADATA_PATH
    
    def _load_specific_documents(self, indices: List[int]) -> Dict[int, dict]:
        """Load only specific documents from metadata"""
        logger.debug(f"Loading {len(indices)} specific documents from metadata")
        
        needed_indices = set(indices)
        results = {}
        
        try:
            with open(self.metadata_path, 'r') as f:
                f.read(1)  # Skip opening bracket
                
                current_idx = 0
                buffer = ""
                depth = 0
                
                for line in f:
                    buffer += line
                    
                    for char in line:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                if current_idx in needed_indices:
                                    try:
                                        doc = json.loads(buffer.strip().rstrip(','))
                                        results[current_idx] = doc
                                        
                                        if len(results) == len(needed_indices):
                                            logger.debug(f"Loaded all {len(results)} documents")
                                            return results
                                    except Exception as e:
                                        logger.warning(f"Error parsing document {current_idx}: {e}")
                                
                                current_idx += 1
                                buffer = ""
                                
                                if current_idx > max(needed_indices):
                                    return results
        
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
        
        logger.debug(f"Loaded {len(results)}/{len(needed_indices)} documents")
        return results
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        
        logger.info(f"Retrieving top {top_k} documents for query: '{query[:50]}...'")

        query_embedding = self.model.encode(query, normalize_embeddings=True)
        query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)
        logger.debug(f"FAISS search returned {len(indices[0])} results")

        documents = self._load_specific_documents(indices[0].tolist())
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            if idx in documents:
                doc = documents[idx]
                results.append({
                    'score': float(score),
                    'question': doc['question'],
                    'chunk': doc['chunk'],
                    'metadata': doc['metadata']
                })
        
        logger.info(f"Retrieved {len(results)} documents successfully")
        return results