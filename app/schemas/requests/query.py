from pydantic import BaseModel, Field

class SearchBodyRequest(BaseModel):
    model: str = Field(default='Text', example="Text")
    value: str = Field(default='Text', example="HCM AI")
