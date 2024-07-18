from fastapi import APIRouter
from models.image import Image
from config.database import collection_name
from schemas.schema import list_serial
from bson import ObjectId

# import torch
# import torch.nn as nn

router = APIRouter()

# GET request
@router.get("/")
async def get_images():
    images = list_serial(collection_name.find())
    # return images
    return {"hello": 'world'}

# POST request for model

# class InputData(BaseModel):
#     x: float
# class PredictionOutput(BaseModel):
#     prediction: float
# @router.post("/predict", response_model=PredictionOutput)
# async def predict(data: InputData):
#     input_tensor = torch.tensor([[data.x]], dtype=torch.float32)
#     with torch.no_grad():
#         prediction = model(input_tensor)
#     return {"prediction": prediction.item()}
