from fastapi import FastAPI
import uvicorn
import joblib

app = FastAPI()


# loading model 
try:
    rf_model = joblib.load('models/Random_Forest.pkl')
    gd_model = joblib.load('models/Gradient_Boosting.pkl')
    xg_model = joblib.load('models/XGBoost.pkl')
except:
    print('model file not found')

# setting baseline model 
class baseline():
    pass



@app.get('/')
def test():
    return {"message":"testing ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)