from fastapi import APIRouter
from config.database import collection_name
from schemas.schema import list_serial
from bson import ObjectId
from aimodels.aimodel import model
from models.model import Image, InputData, PredictionOutput
import torch

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
async def upload_image(image: Image):
    collection_name.insert_one(dict(image))