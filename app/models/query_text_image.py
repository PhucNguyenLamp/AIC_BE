from typing import Optional
from beanie import Document
from pydantic import Field
from bson import ObjectId


class TextImage(Document):
    # Beanie automatically uses ObjectId for the '_id' field
    index: int = Field(default=0)
    value: Optional[str] = None

    class Settings:
        # No need to define custom indexes for ObjectId; it's automatically indexed
        pass
