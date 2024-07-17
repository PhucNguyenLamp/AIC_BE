from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import torch
from transformers import CLIPProcessor, CLIPModel
import os
from PIL import Image
import numpy as np

app = FastAPI()

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['image_db']
collection = db['images']

class Query(BaseModel):
    text: str

@app.post("/search")
async def search_images(query: Query):
    # Generate text embedding
    inputs = processor(text=[query.text], return_tensors="pt", padding=True, truncation=True)
    text_features = model.get_text_features(**inputs)
    text_embedding = text_features.detach().numpy().flatten()

    # Perform similarity search in MongoDB
    results = collection.aggregate([
        {
            "$addFields": {
                "similarity": {
                    "$dotProduct": ["$embedding", text_embedding.tolist()]
                }
            }
        },
        {"$sort": {"similarity": -1}},
        {"$limit": 10}
    ])

    # Prepare results
    image_results = []
    for r in results:
        image_results.append({
            "filename": r["filename"],
            "path": r["path"],
            "similarity": r["similarity"]
        })

    return {"results": image_results}

@app.on_event("startup")
async def startup_event():
    # This function could be used to load images and generate embeddings if needed
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
