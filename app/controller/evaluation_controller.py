from typing import Dict, Any, Optional
from app.services.evaluation_service import EvaluationService
from app.utils.logger import get_logger

logger = get_logger(__name__)

class EvaluationController:
    """Controller for evaluation operations"""
    
    def __init__(self):
        self.evaluation_service = EvaluationService()
    
    async def run_evaluation(
        self,
        test_file: str = "evaluation/test_dataset.json",
        top_k: int = 5,
        max_samples: Optional[int] = None,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation and return results
        
        Args:
            test_file: Path to test dataset
            top_k: Number of documents to retrieve
            max_samples: Limit samples for quick test
            save_report: Whether to save report to file
        
        Returns:
            Evaluation report
        """
        try:
            logger.info("Starting evaluation...")
            
            report = await self.evaluation_service.run_evaluation(
                test_file=test_file,
                top_k=top_k,
                max_samples=max_samples
            )
            
            if save_report:
                timestamp = report['metadata']['timestamp'].replace(':', '-')
                output_path = f"evaluation/reports/eval_report_{timestamp}.json"
                self.evaluation_service.save_report(report, output_path)
            
            return report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise