import pickle
import pandas as pd
import os

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

def predict_species(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return int(prediction)