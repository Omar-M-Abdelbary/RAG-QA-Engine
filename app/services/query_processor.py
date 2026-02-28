import re
from typing import List, Dict

class QueryProcessor:
    """
    Process and enhance user queries for better retrieval
    
    Responsibilities:
    - Clean and normalize queries
    - Expand queries with synonyms/variations
    - Handle multi-turn conversations
    - Query rewriting
    """
    
    def __init__(self):
        # Common words to remove (stop words for query expansion)
        self.stop_words = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'when', 'where', 'who'}
        
        # Query expansion patterns (simple synonyms)
        self.expansions = {
            'capital': ['capital', 'capital city', 'main city'],
            'invented': ['invented', 'created', 'discovered', 'made'],
            'work': ['work', 'function', 'operate'],
        }
    
    def clean_query(self, query: str) -> str:
        """
        Clean and normalize query text
        
        Args:
            query: Raw user query
            
        Returns:
            Cleaned query
        """
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove special characters except question marks and apostrophes
        query = re.sub(r'[^\w\s\?\']', '', query)
        
        return query
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate query variations for better retrieval
        
        Args:
            query: Cleaned query
            
        Returns:
            List of query variations
        """
        variations = [query]
        
        # Check for expandable terms
        words = query.split()
        for word in words:
            if word in self.expansions:
                for synonym in self.expansions[word]:
                    if synonym != word:
                        # Create variation with synonym
                        new_query = query.replace(word, synonym)
                        variations.append(new_query)
        
        return variations[:3]  # Limit to 3 variations
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query
        
        Args:
            query: Query text
            
        Returns:
            List of keywords
        """
        words = query.lower().split()
        
        # Remove stop words
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        return keywords
    
    def process_query(self, query: str, context: List[Dict] = None) -> Dict:
        """
        Main processing pipeline for queries
        
        Args:
            query: Raw user query
            context: Previous conversation context (for multi-turn)
            
        Returns:
            Processed query information
        """
        # Step 1: Clean
        cleaned = self.clean_query(query)
        
        # Step 2: Extract keywords
        keywords = self.extract_keywords(cleaned)
        
        # Step 3: Generate variations
        variations = self.expand_query(cleaned)
        
        # Step 4: Handle multi-turn context (if provided)
        if context:
            # Add context from previous questions
            context_keywords = []
            for prev in context[-2:]:  # Last 2 exchanges
                if 'question' in prev:
                    prev_keywords = self.extract_keywords(prev['question'])
                    context_keywords.extend(prev_keywords)
            
            # Merge with current keywords (remove duplicates)
            keywords = list(set(keywords + context_keywords))
        
        return {
            'original': query,
            'cleaned': cleaned,
            'keywords': keywords,
            'variations': variations,
            'has_context': context is not None
        }