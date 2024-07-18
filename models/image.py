from pydantic import BaseModel

class Image(BaseModel):
    path: str
    value: float
    