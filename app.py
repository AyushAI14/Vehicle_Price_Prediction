from fastapi import FastAPI, Form, Request
import uvicorn
import joblib
from pydantic import BaseModel
import os
import pandas as pd
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Only load XGBoost model
MODEL_PATH = 'models/XGBoost.pkl'

if os.path.exists(MODEL_PATH):
    xg_model = joblib.load(MODEL_PATH)
    print("XGBoost model loaded successfully.")
else:
    print(f"Model file not found at {MODEL_PATH}")
    xg_model = None

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')


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


@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
        "index.html", {"request": request, "predicted_price": None}
    )


@app.post("/predict")
async def price_predict(
    request: Request,
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    engine: str = Form(...),
    cylinders: float = Form(...),
    fuel: str = Form(...),
    mileage: float = Form(...),
    transmission: str = Form(...),
    trim: str = Form(...),
    body: str = Form(...),
    doors: float = Form(...),
    exterior_color: str = Form(...),
    interior_color: str = Form(...),
    drivetrain: str = Form(...),
):
    if xg_model is None:
        return {"error": "Model not loaded properly."}

    input_data = pd.DataFrame(
        [{
            'make': make,
            'model': model,
            'year': year,
            'engine': engine,
            'cylinders': cylinders,
            'fuel': fuel,
            'mileage': mileage,
            'transmission': transmission,
            'trim': trim,
            'body': body,
            'doors': doors,
            'exterior_color': exterior_color,
            'interior_color': interior_color,
            'drivetrain': drivetrain
        }]
    )

    prediction = xg_model.predict(input_data)[0]

    return templates.TemplateResponse("index.html",{"request": request,"predicted_price": float(prediction)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
