import time
from typing import Dict
from collections import defaultdict
from datetime import datetime
from app.utils.logger import get_logger

logger = get_logger(__name__)

class PerformanceMonitor:
    """
    Monitor and track system performance metrics
    
    Tracks:
    - Request latency
    - Cache hit rates
    - LLM call counts
    - Error rates
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.start_time = time.time()
        
        logger.info("Performance Monitor initialized")
    
    def record_latency(self, operation: str, duration: float):
        """
        Record operation latency
        
        Args:
            operation: Operation name (e.g., 'retrieval', 'generation', 'total')
            duration: Duration in seconds
        """
        self.metrics[f"{operation}_latency"].append(duration)
        logger.debug(f"{operation} latency: {duration:.3f}s")
    
    def increment_counter(self, counter: str):
        """
        Increment a counter
        
        Args:
            counter: Counter name (e.g., 'cache_hits', 'llm_calls')
        """
        self.counters[counter] += 1
    
    def record_error(self, error_type: str):
        """Record an error occurrence"""
        self.counters[f"error_{error_type}"] += 1
        logger.warning(f"Error recorded: {error_type}")
    
    def get_stats(self) -> Dict:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'uptime_seconds': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat(),
            'counters': dict(self.counters),
            'latencies': {}
        }
        
        for key, values in self.metrics.items():
            if values:
                stats['latencies'][key] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'total': sum(values)
                }
        

        total_requests = self.counters.get('total_requests', 0)
        cache_hits = self.counters.get('cache_hits', 0)
        
        if total_requests > 0:
            stats['cache_hit_rate'] = cache_hits / total_requests
            stats['avg_requests_per_minute'] = total_requests / (stats['uptime_seconds'] / 60)
        
        return stats
    
    def log_stats(self):
        """Log current statistics"""
        stats = self.get_stats()
        
        logger.info("=== Performance Statistics ===")
        logger.info(f"Uptime: {stats['uptime_seconds']:.0f}s")
        logger.info(f"Total requests: {self.counters.get('total_requests', 0)}")
        logger.info(f"Cache hits: {self.counters.get('cache_hits', 0)}")
        logger.info(f"LLM calls: {self.counters.get('llm_calls', 0)}")
        
        if 'cache_hit_rate' in stats:
            logger.info(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
        
        # Log latencies
        for op, metrics in stats.get('latencies', {}).items():
            logger.info(f"{op}: avg={metrics['avg']:.3f}s, min={metrics['min']:.3f}s, max={metrics['max']:.3f}s")
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()
        self.start_time = time.time()
        logger.info("Performance metrics reset")

# Global instance
_monitor = None

def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor