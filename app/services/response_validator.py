import re
from typing import Dict, List

class ResponseValidator:
    """
    Validate and filter LLM responses for quality
    
    Responsibilities:
    - Check if response is relevant
    - Detect hallucinations
    - Validate answer completeness
    - Provide quality score
    """
    
    def __init__(self):
        # Phrases indicating uncertainty/lack of info
        self.uncertainty_phrases = [
            "i don't have",
            "i don't know",
            "cannot answer",
            "not enough information",
            "insufficient information",
            "i'm not sure",
            "unable to find"
        ]
        
        # Phrases indicating hallucination/external knowledge
        self.hallucination_indicators = [
            "as everyone knows",
            "it is common knowledge",
            "generally speaking",
            "in my experience"
        ]
    
    def check_relevance(self, question: str, answer: str, contexts: List[str]) -> bool:
        """
        Check if answer is relevant to question and contexts
        
        Args:
            question: User's question
            answer: Generated answer
            contexts: Retrieved contexts
            
        Returns:
            True if relevant, False otherwise
        """
        # Basic check: answer should not be too short
        if len(answer.strip()) < 10:
            return False
        
        # Check if answer mentions key terms from question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # At least some overlap
        overlap = question_words.intersection(answer_words)
        
        return len(overlap) > 0
    
    def detect_uncertainty(self, answer: str) -> bool:
        """
        Detect if answer expresses uncertainty
        
        Args:
            answer: Generated answer
            
        Returns:
            True if uncertain, False otherwise
        """
        answer_lower = answer.lower()
        
        for phrase in self.uncertainty_phrases:
            if phrase in answer_lower:
                return True
        
        return False
    
    def detect_hallucination(self, answer: str, contexts: List[str]) -> bool:
        """
        Detect potential hallucinations
        
        Args:
            answer: Generated answer
            contexts: Retrieved contexts
            
        Returns:
            True if potential hallucination detected
        """
        answer_lower = answer.lower()
        
        # Check for hallucination indicator phrases
        for phrase in self.hallucination_indicators:
            if phrase in answer_lower:
                return True
        
        # Check if answer contains information not in contexts
        # (Simple heuristic: check for specific numbers/dates not in context)
        answer_numbers = re.findall(r'\b\d{4}\b', answer)  # Years
        context_text = ' '.join(contexts).lower()
        
        for num in answer_numbers:
            if num not in context_text:
                # Number in answer but not in context - potential hallucination
                return True
        
        return False
    
    def calculate_quality_score(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str],
        retrieval_scores: List[float]
    ) -> float:
        """
        Calculate overall quality score for response
        
        Args:
            question: User's question
            answer: Generated answer
            contexts: Retrieved contexts
            retrieval_scores: Similarity scores from retrieval
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        score = 1.0
        
        # Factor 1: Relevance (0.3 weight)
        if not self.check_relevance(question, answer, contexts):
            score -= 0.3
        
        # Factor 2: Uncertainty (-0.2)
        if self.detect_uncertainty(answer):
            score -= 0.2
        
        # Factor 3: Hallucination (-0.4)
        if self.detect_hallucination(answer, contexts):
            score -= 0.4
        
        # Factor 4: Answer length (ideal: 50-500 chars)
        answer_len = len(answer)
        if answer_len < 20:
            score -= 0.2
        elif answer_len > 1000:
            score -= 0.1
        
        # Factor 5: Retrieval confidence (average of top 3 scores)
        if retrieval_scores:
            avg_retrieval_score = sum(retrieval_scores[:3]) / min(3, len(retrieval_scores))
            if avg_retrieval_score < 0.5:
                score -= 0.2
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    def validate_response(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str],
        retrieval_scores: List[float]
    ) -> Dict:
        """
        Complete validation pipeline
        
        Args:
            question: User's question
            answer: Generated answer
            contexts: Retrieved contexts
            retrieval_scores: Similarity scores
            
        Returns:
            Validation results
        """
        quality_score = self.calculate_quality_score(
            question, answer, contexts, retrieval_scores
        )
        
        return {
            'is_valid': quality_score >= 0.5,
            'quality_score': quality_score,
            'is_relevant': self.check_relevance(question, answer, contexts),
            'has_uncertainty': self.detect_uncertainty(answer),
            'potential_hallucination': self.detect_hallucination(answer, contexts),
            'recommendation': 'accept' if quality_score >= 0.7 else 'review' if quality_score >= 0.5 else 'reject'
        }