import torch
import torch.nn as nn
from pydantic import BaseModel

class InputData(BaseModel):
    x: float
class PredictionOutput(BaseModel):
    prediction: float

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
model.load_state_dict(torch.load("./aimodels/linear_regression_model.pth"))
model.eval()  # Set the model to evaluation mode

