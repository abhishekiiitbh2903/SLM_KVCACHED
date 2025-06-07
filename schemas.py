from pydantic import BaseModel
from enum import Enum

class ModelType(str, Enum):
    UNTRAINED= "untrained"
    BASIC = "basic"
    CACHED = "cached"

class PredictRequest(BaseModel):
    input_text: str
    model_type: ModelType

class PredictResponse(BaseModel):
    result: str
    time: float


