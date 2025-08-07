from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from app.predict import predict_species
from train_model import train_model

app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "Iris Prediction API is running."}

@app.post("/predict")
def predict(data: IrisInput):
    result = predict_species(data.dict())
    return {"prediction": result}

@app.post("/retrain")
def retrain(x_api_key: str = Header(...)):
    if x_api_key != "my-secret-key":
        raise HTTPException(status_code=403, detail="Unauthorized")
    train_model()
    return {"message": "Model retrained and updated."}
