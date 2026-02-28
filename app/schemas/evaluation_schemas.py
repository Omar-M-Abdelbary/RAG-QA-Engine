from pydantic import BaseModel, Field
from typing import Optional

class EvaluationRequest(BaseModel):
    test_file: str = Field(
        default="evaluation/test_dataset.json",
        description="Path to test dataset"
    )
    top_k: int = Field(default=5, ge=1, le=10)
    max_samples: Optional[int] = Field(
        default=None,
        description="Limit samples for quick test (e.g., 10 for testing)"
    )
    save_report: bool = Field(default=True)
