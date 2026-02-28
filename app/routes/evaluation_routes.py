from fastapi import APIRouter, HTTPException

from app.controller.evaluation_controller import EvaluationController
from app.schemas.evaluation_schemas import EvaluationRequest
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

evaluation_controller = EvaluationController()

@router.post("/run", tags=["Evaluation"])
async def run_evaluation(request: EvaluationRequest):
    """
    Run evaluation on test dataset
    
    Example:
```json
    {
        "test_file": "evaluation/test_dataset.json",
        "top_k": 5,
        "max_samples": 10,
        "save_report": true
    }
```
    """
    try:
        report = await evaluation_controller.run_evaluation(
            test_file=request.test_file,
            top_k=request.top_k,
            max_samples=request.max_samples,
            save_report=request.save_report
        )
        
        # Return summary (not full detailed results to avoid large response)
        return {
            "status": "success",
            "metadata": report['metadata'],
            "metrics": report['metrics'],
            "sample_failures": report['failed_questions'][:5]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))