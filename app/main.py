from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_species

app = FastAPI()

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: IrisData):
    result = predict_species(data.dict())
    return {"prediction": result}