from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import os
import pandas as pd

app = FastAPI()

# Only load XGBoost model
MODEL_PATH = 'models/XGBoost.pkl'

if os.path.exists(MODEL_PATH):
    xg_model = joblib.load(MODEL_PATH)
    print("XGBoost model loaded successfully.")
else:
    print(f"Model file not found at {MODEL_PATH}")
    xg_model = None

# Define input features schema
class CarFeatures(BaseModel):
    make: str
    model: str
    year: int
    engine: str
    cylinders: float
    fuel: str
    mileage: float
    transmission: str
    trim: str
    body: str
    doors: float
    exterior_color: str
    interior_color: str
    drivetrain: str

@app.get('/')
def test():
    return {"message": "API is working fine."}

@app.post('/predict')
def price_predict(features: CarFeatures):
    if xg_model is None:
        return {"error": "Model not loaded properly."}

    # input_data = [[
    #     features.make,
    #     features.model,
    #     features.year,
    #     features.engine,
    #     features.cylinders,
    #     features.fuel,
    #     features.mileage,
    #     features.transmission,
    #     features.trim,
    #     features.body,
    #     features.doors,
    #     features.exterior_color,
    #     features.interior_color,
    #     features.drivetrain
    # ]]
    input_data = pd.DataFrame([features.dict()])

    prediction = xg_model.predict(input_data)[0]
    return {'predicted_price': float(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
