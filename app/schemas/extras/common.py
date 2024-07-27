from typing import Generic, TypeVar
from pydantic import Field
from pydantic.generics import GenericModel

# Create a type variable that can be any type
T = TypeVar('T')

class Response(GenericModel, Generic[T]):
    data: T
    message: str = Field(example="Success", default="Success")
    status_code: int = Field( example=200, default=200)