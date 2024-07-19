from pydantic import BaseModel

class Image(BaseModel):
    path: str
    value: float
class InputData(BaseModel):
    x: float
class PredictionOutput(BaseModel):
    prediction: float
