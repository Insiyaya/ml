import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import mlflow
import mlflow.sklearn

def train_model():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    model = RandomForestClassifier()
    model.fit(X, y)

    with open("app/model.pkl", "wb") as f:
        pickle.dump(model, f)

    mlflow.set_experiment("iris_model_experiment")
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", model.score(X, y))

    print(" Model trained and logged with MLflow.")

if __name__ == "__main__":
    train_model()
