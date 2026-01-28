from pydantic import BaseModel
from typing import List

class TrainRequest(BaseModel):
    X: List[List[float]]
    y: List[float]

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: float


