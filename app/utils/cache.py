import hashlib
import json
import time
from typing import Optional, Dict, Any
from pathlib import Path
from app.utils.logger import get_logger

logger = get_logger(__name__)

class RAGCache:
    """
    Simple file-based cache for RAG responses
    
    Benefits:
    - Avoid redundant LLM calls
    - Faster responses for repeated questions
    - Save API costs
    """
    
    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl
        
        logger.info(f"Cache initialized in {self.cache_dir} with TTL={ttl}s")
    
    def _generate_key(self, question: str, top_k: int) -> str:
        """
        Generate cache key from question and parameters
        
        Args:
            question: User question
            top_k: Number of contexts
            
        Returns:
            Cache key (hash)
        """
        # Normalize question (lowercase, strip)
        normalized = question.lower().strip()
        
        # Create key from question + top_k
        cache_input = f"{normalized}_{top_k}"
        
        # Hash for filename safety
        key = hashlib.md5(cache_input.encode()).hexdigest()
        
        return key
    
    def get(self, question: str, top_k: int) -> Optional[Dict[str, Any]]:
        """
        Get cached response if exists and not expired
        
        Args:
            question: User question
            top_k: Number of contexts
            
        Returns:
            Cached response or None
        """
        key = self._generate_key(question, top_k)
        cache_file = self.cache_dir / f"{key}.json"
        
        # Check if cache file exists
        if not cache_file.exists():
            logger.debug(f"Cache miss for key: {key}")
            return None
        
        try:
            # Load cache
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Check if expired
            cached_time = cached.get('timestamp', 0)
            age = time.time() - cached_time
            
            if age > self.ttl:
                logger.debug(f"Cache expired for key: {key} (age: {age:.0f}s)")
                cache_file.unlink()  # Delete expired cache
                return None
            
            logger.info(f"Cache hit for key: {key} (age: {age:.0f}s)")
            return cached.get('data')
            
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, question: str, top_k: int, data: Dict[str, Any]):
        """
        Store response in cache
        
        Args:
            question: User question
            top_k: Number of contexts
            data: Response data to cache
        """
        key = self._generate_key(question, top_k)
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            # Prepare cache entry
            cache_entry = {
                'timestamp': time.time(),
                'question': question,
                'top_k': top_k,
                'data': data
            }
            
            # Write to file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Cached response for key: {key}")
            
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def clear(self):
        """Clear all cache files"""
        try:
            count = 0
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                count += 1
            
            logger.info(f"Cleared {count} cache entries")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total = len(cache_files)
            
            # Check how many are expired
            expired = 0
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                    
                    age = time.time() - cached.get('timestamp', 0)
                    if age > self.ttl:
                        expired += 1
                except:
                    pass
            
            return {
                'total_entries': total,
                'expired_entries': expired,
                'valid_entries': total - expired,
                'cache_dir': str(self.cache_dir),
                'ttl': self.ttl
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}