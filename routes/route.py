from fastapi import APIRouter
from config.database import collection_name
from schemas.schema import list_serial
from bson import ObjectId
from aimodels.aimodel import model, PredictionOutput, InputData
import torch

router = APIRouter()

# GET request
@router.get("/")
async def get_images():
    images = list_serial(collection_name.find())
    # return images
    return {"hello": 'world'}

# POST request for model
@router.post("/predict", response_model=PredictionOutput)
async def predict(data: InputData):
    input_tensor = torch.tensor([[data.x]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)
    return {"prediction": prediction.item()}
