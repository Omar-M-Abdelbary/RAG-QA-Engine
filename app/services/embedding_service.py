from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from app.core.config import Config

class EmbeddingService:
    """Handle text embeddings using SentenceTransformers"""
    
    def __init__(self):
        self.config = Config()
        print(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        print(f" Model loaded! Embedding dimension: {self.config.EMBEDDING_DIMENSION}")
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Create embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dimension)
        """
        print(f"Creating embeddings for {len(texts)} texts...")
        
        # Create embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.BATCH_SIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        print(f" Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy array of shape (embedding_dimension,)
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding