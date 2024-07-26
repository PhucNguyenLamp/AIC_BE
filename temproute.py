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
import json 
import random
import threading


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
    model_mode: List[str] = Field(..., description="List of model types to use for querying")
    data: Optional[List[str]] = Field(None, description="List of text queries/images")
    based_history: Optional[List[str]] = Field(None, description="List of video IDs for history-based querying")

class KeyframeData(BaseModel):
    id: str # mongodb id
    video_id: str # V001
    keyframe_index: int # 000000

class QueryResponse(BaseModel):
    data: List[KeyframeData]
    total: int

@router.post("/api/query", response_model=QueryResponse)
async def query_keyframes(
    body: QueryBody = Body(...), # request body
    page: int = Query(1, description="Page number"), # ?page=10 tren url
    limit: int = Query(10, description="Number of items per page")
):
    # model_query = QueryHandlingModel()
    # model_mode, text, image, based_history = body.model, body.data, body.image, body.based_history
    # result = model_query.inference(
    #      model_mode,# text, image, [text, image, audio, ocr],
    #      data, # ["text", "text", "text", "text" ]
    #    
    #) : [tuple of (keyframe_index, score)], keyframe_index: [int], score: [float]
    # [tuple(text), tuple(image), None, None]

    # destructure body
    # model, data, image, based_history = body.model, body.data, body.image, body.based_history
    
    
    
    #load json file for displaying
    with open ('/json/global_index2kf_path.json') as f:
        video_path_json = json.load(f)
    # search dict -> keyframedata array
    
    return QueryResponse(
        data=[
            KeyframeData(
                id="123",
                keyframe_index=1.02,
                video_id="bruh"
            )
        ],
        total=1 # len(array)
    )
