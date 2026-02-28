import re
from typing import Tuple, List, Dict, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)

class PreprocessingService:
    """Service for text preprocessing"""
    
    def preprocess(
        self, 
        text: str, 
        options: Dict[str, Any] = None
    ) -> Tuple[str, List[str]]:
        """
        Preprocess text with various operations
        
        Args:
            text: Input text
            options: Dictionary of preprocessing options
        
        Returns:
            Tuple of (processed_text, operations_applied)
        """
        if options is None:
            options = {}
        
        operations_applied = []
        processed_text = text
        
        # Lowercase
        if options.get('lowercase', False):
            processed_text = processed_text.lower()
            operations_applied.append('lowercase')
        
        # Remove special characters
        if options.get('remove_special_chars', False):
            processed_text = re.sub(r'[^\w\s]', '', processed_text)
            operations_applied.append('remove_special_chars')
        
        # Remove numbers
        if options.get('remove_numbers', False):
            processed_text = re.sub(r'\d+', '', processed_text)
            operations_applied.append('remove_numbers')
        
        # Remove extra spaces
        if options.get('remove_extra_spaces', True):
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            operations_applied.append('remove_extra_spaces')
        
        return processed_text, operations_applied