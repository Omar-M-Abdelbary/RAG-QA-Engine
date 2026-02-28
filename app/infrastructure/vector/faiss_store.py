import faiss
import numpy as np
import json
import os
from typing import List, Tuple
from app.core.config import Config

class VectorStore:
    """Manage FAISS vector database"""
    
    def __init__(self):
        self.config = Config()
        self.index = None
        self.metadata = []
    
    def create_index(self, dimension: int):
        """
        Create a new FAISS index
        
        Args:
            dimension: Embedding dimension (384 for our model)
        """
        print(f"Creating FAISS index with dimension: {dimension}")
        
        # Use IndexFlatIP for cosine similarity (Inner Product)
        # Since embeddings are normalized, IP = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        
        print(f" FAISS index created")
    
    def add_embeddings(self, embeddings: np.ndarray, documents: List[dict]):
        """
        Add embeddings to the index
        
        Args:
            embeddings: numpy array of shape (n, dimension)
            documents: List of document dictionaries with metadata
        """
        if self.index is None:
            raise ValueError("Index not created! Call create_index() first")
        
        print(f" Adding {len(embeddings)} embeddings to FAISS index...")

        self.index.add(embeddings)
        self.metadata.extend(documents)
        
        print(f" Added {len(embeddings)} embeddings. Total in index: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[float], List[dict]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            (scores, documents) - Lists of similarity scores and matching documents
        """
        if self.index is None:
            raise ValueError("Index not loaded!")
        
        query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        
        return scores[0].tolist(), results
    
    def save(self):
        """Save index and metadata to disk"""
        if self.index is None:
            raise ValueError("No index to save!")
        
        os.makedirs(os.path.dirname(self.config.FAISS_INDEX_PATH), exist_ok=True)
        
        print(f" Saving FAISS index to {self.config.FAISS_INDEX_PATH}...")
        faiss.write_index(self.index, self.config.FAISS_INDEX_PATH)
        
        print(f" Saving metadata to {self.config.FAISS_METADATA_PATH}...")
        with open(self.config.FAISS_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        
        print(f" Saved index with {self.index.ntotal} vectors")
    
    def load(self):
        """Load index and metadata from disk"""
        print(f" Loading FAISS index from {self.config.FAISS_INDEX_PATH}...")
        self.index = faiss.read_index(self.config.FAISS_INDEX_PATH)
        
        print(f" Loading metadata from {self.config.FAISS_METADATA_PATH}...")
        with open(self.config.FAISS_METADATA_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f" Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")