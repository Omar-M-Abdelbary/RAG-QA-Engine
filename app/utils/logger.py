import logging
import sys
from pathlib import Path
from datetime import datetime

class RAGLogger:
    """
    Centralized logging utility for RAG system
    
    Usage:
        from app.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Message here")
    """
    
    _loggers = {}
    _log_dir = None
    
    @classmethod
    def setup(cls, log_dir: str = "logs", level: str = "INFO"):
        """
        Setup logging configuration (call once at app startup)
        
        Args:
            log_dir: Directory for log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(exist_ok=True)
        
        # Create log filename with timestamp
        log_file = cls._log_dir / f"rag_system_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                # Console handler
                logging.StreamHandler(sys.stdout),
                # File handler
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get logger instance for a module
        
        Args:
            name: Logger name (usually __name__ of the module)
            
        Returns:
            Logger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]

# Convenience function
def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance
    
    Usage:
        from app.utils.logger import get_logger
        logger = get_logger(__name__)
    """
    return RAGLogger.get_logger(name)

# Initialize logging on import
RAGLogger.setup()