from typing import List, Dict, Any
import json
from pathlib import Path
from datetime import datetime
import time
from difflib import SequenceMatcher

from app.services.rag_service import RAGService
from app.utils.logger import get_logger

logger = get_logger(__name__)

class EvaluationService:
    """Service for evaluating RAG system performance"""
    
    def __init__(self):
        self.rag_service = None
    
    def load_test_dataset(self, test_file: str) -> List[Dict[str, Any]]:
        """Load test dataset from file"""
        logger.info(f"Loading test dataset from: {test_file}")
        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        logger.info(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using SequenceMatcher
        
        Returns:
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def evaluate_answer(
        self, 
        generated_answer: str, 
        ground_truth: str,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate a single answer against ground truth
        
        Args:
            generated_answer: Answer from RAG system
            ground_truth: Correct answer from dataset
            threshold: Similarity threshold for "correct"
        
        Returns:
            Evaluation metrics for this answer
        """
        similarity = self.calculate_similarity(generated_answer, ground_truth)
        
        is_correct = similarity >= threshold

        gt_words = set(ground_truth.lower().split())
        gen_words = set(generated_answer.lower().split())
        
        keyword_overlap = len(gt_words & gen_words) / len(gt_words) if gt_words else 0
        
        return {
            "similarity_score": round(similarity, 3),
            "keyword_overlap": round(keyword_overlap, 3),
            "is_correct": is_correct,
            "generated_length": len(generated_answer),
            "ground_truth_length": len(ground_truth)
        }
    
    async def run_evaluation(
        self, 
        test_file: str,
        top_k: int = 5,
        max_samples: int = None
    ) -> Dict[str, Any]:
        """
        Run full evaluation on test dataset
        
        Args:
            test_file: Path to test dataset JSON
            top_k: Number of documents to retrieve
            max_samples: Limit number of test samples (for quick testing)
        
        Returns:
            Complete evaluation results
        """
        logger.info("=" * 60)
        logger.info(" Starting RAG System Evaluation")
        logger.info("=" * 60)
        
        test_data = self.load_test_dataset(test_file)
        
        if max_samples:
            test_data = test_data[:max_samples]
            logger.info(f" Limiting to {max_samples} samples for quick test")
  
        logger.info(" Initializing RAG Service...")
        if self.rag_service is None:
            self.rag_service = RAGService()
    
        results = []
        correct_count = 0
        total_latency = 0
        failed_questions = []
        
        for idx, sample in enumerate(test_data, 1):
            question = sample['question']
            ground_truth = sample['answer']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {idx}/{len(test_data)}: {question[:50]}...")
            
            try:
                start_time = time.time()
                
                
                rag_response = await self.rag_service.answer_question(
                    question=question,
                    top_k=top_k
                )
                
                latency = time.time() - start_time
                total_latency += latency
                
                generated_answer = rag_response.get('answer', '')
                
                
                eval_result = self.evaluate_answer(generated_answer, ground_truth)
                
                
                result = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_answer": generated_answer,
                    "latency_seconds": round(latency, 3),
                    "retrieved_docs": len(rag_response.get('retrieved_documents', [])),
                    **eval_result
                }
                
                results.append(result)
                
                if eval_result['is_correct']:
                    correct_count += 1
                    logger.info(f" CORRECT (similarity: {eval_result['similarity_score']})")
                else:
                    logger.info(f" INCORRECT (similarity: {eval_result['similarity_score']})")
                    failed_questions.append({
                        "question": question,
                        "similarity": eval_result['similarity_score']
                    })
                
            except Exception as e:
                logger.error(f" Error processing question: {e}")
                failed_questions.append({
                    "question": question,
                    "error": str(e)
                })
        
        
        accuracy = correct_count / len(test_data) if test_data else 0
        avg_latency = total_latency / len(test_data) if test_data else 0
        avg_similarity = sum(r['similarity_score'] for r in results) / len(results) if results else 0
        avg_keyword_overlap = sum(r['keyword_overlap'] for r in results) / len(results) if results else 0
        
        
        evaluation_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_dataset": test_file,
                "total_samples": len(test_data),
                "top_k": top_k
            },
            "metrics": {
                "accuracy": round(accuracy, 3),
                "correct_answers": correct_count,
                "incorrect_answers": len(test_data) - correct_count,
                "average_similarity": round(avg_similarity, 3),
                "average_keyword_overlap": round(avg_keyword_overlap, 3),
                "average_latency_seconds": round(avg_latency, 3),
                "total_time_seconds": round(total_latency, 2)
            },
            "failed_questions": failed_questions[:10],  # Top 10 failures
            "detailed_results": results
        }
        
        logger.info("\n" + "=" * 60)
        logger.info(" EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f" Accuracy: {accuracy*100:.1f}%")
        logger.info(f" Avg Similarity: {avg_similarity:.3f}")
        logger.info(f" Avg Latency: {avg_latency:.3f}s")
        logger.info("=" * 60)
        
        return evaluation_report
    
    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save evaluation report to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f" Evaluation report saved: {output_file}")