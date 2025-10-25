

# src/predictor.py
import joblib
import pands as pd

def predict_species(features, model_path):
    """
    features: list of 4 numeric values (sepal length, sepal width, petal length, petal width)
    """
    model = joblib.load(model_path)
    prediction = model.predict([features])[0]
    return prediction
