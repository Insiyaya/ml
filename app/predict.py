import pickle
import pandas as pd

with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_species(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return int(prediction)
