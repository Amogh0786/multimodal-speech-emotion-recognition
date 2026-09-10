from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    distribution: Dict[str, float]
    processing_latency_ms: float
