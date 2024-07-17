from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
model.load_state_dict(torch.load("linear_regression_model.pth"))
model.eval()  # Set the model to evaluation mode
class InputData(BaseModel):
    x: float

class PredictionOutput(BaseModel):
    prediction: float

@app.get("/")
async def root():
    return {"message": "Hello World"}
@app.post("/predict", response_model=PredictionOutput)
async def predict(data: InputData):
    input_tensor = torch.tensor([[data.x]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)
    return {"prediction": prediction.item()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
