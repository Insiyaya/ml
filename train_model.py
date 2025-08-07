import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle
import os
import mlflow
import mlflow.sklearn

def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    os.makedirs("app", exist_ok=True)
    with open("app/model.pkl", "wb") as f:
        pickle.dump(model, f)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("iris_classifier")

    with mlflow.start_run():
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_metric("accuracy", model.score(X, y))
        mlflow.sklearn.log_model(model, "model")

    print("Model trained and logged with MLflow")

if __name__ == "__main__":
    train_model()