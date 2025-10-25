
# src/predictor.py
import joblib
from sklearn.datasets import load_iris

def predict_species(features, model_path):
    """
    features: list of 4 numeric values (sepal length, sepal width, petal length, petal width)
    """
    model = joblib.load(model_path)
    iris = load_iris()
    prediction = model.predict([features])[0]
    species = iris.target_names[prediction]
    return species
