
# main.py
from src.model_trainer import train_model
from src.predictor import predict_species
from sklearn.datasets import load_iris
import pandas as pd

def main():
    print("🌸 IRIS FLOWER CLASSIFIER 🌸")

    # Load dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target

    # Train and save model
    model_path = train_model(df)

    # Test prediction
    sample = [5.1, 3.5, 1.4, 0.2]  # Example: Iris-setosa
    prediction = predict_species(sample, model_path)
    print(f"Prediction for sample {sample}: {prediction}")

if __name__ == "__main__":
    main()
