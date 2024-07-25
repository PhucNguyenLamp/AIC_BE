from fastapi import APIRouter
from config.database import collection_name
from schemas.schema import list_serial
from bson import ObjectId
from aimodels.aimodel import model
from models.model import Image, InputData, PredictionOutput
import torch
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import Body, Query, Path, HTTPException

router = APIRouter()

# GET request
@router.get("/")
async def get_images():
    images = list_serial(collection_name.find())
    # return images
    return images

# POST request for model
@router.post("/predict", response_model=PredictionOutput)
async def predict(data: InputData):
    input_tensor = torch.tensor([[data.x]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)
    return {"prediction": prediction.item()}

@router.post('/uploadsomething')
async def upload_image(image:Image):
    collection_name.insert_one(dict(image))

class QueryBody(BaseModel):
    model: List[str] = Field(..., description="List of model types to use for querying")
    data: Optional[List[str]] = Field(None, description="List of text queries")
    image: Optional[str] = Field(None, description="Base64 encoded image")
    based_history: Optional[List[str]] = Field(None, description="List of video IDs for history-based querying")

class KeyframeData(BaseModel):
    id: str
    timestamp: float
    video_id: str

class QueryResponse(BaseModel):
    data: List[KeyframeData]
    total: int

@router.post("/api/{video_id}/query", response_model=QueryResponse)
async def query_keyframes(
    video_id: str = Path(..., description="Video ID (e.g., v1, v2, v3)"),
    body: QueryBody = Body(...),
    page: int = Query(1, description="Page number"),
    limit: int = Query(10, description="Number of items per page")
):
    video_folder = f"/{video_id}"
    # destructure body
    model, data, image, based_history = body.model, body.data, body.image, body.based_history

    # choose model to run
    for m in model:
        if m == "text":
            print("run text model")
        elif m == "image":
            print("run image model")
    
    return QueryResponse(
        data=[
            KeyframeData(
                id="123",
                timestamp=1.02,
                video_id="bruh"
            )
        ],
        total=1
    )
