

# main.py
from src.model_trainer import train_model
from src.predictor import predict_species
from sklearn.datasets import load_iris
import pandas as pd

def main():
    print("🌸 IRIS FLOWER CLASSIFIER 🌸")

    # Load dataset
    df = pd.read_csv("music.csv")

    # Train and save model
    model_path = train_model(df)

    # Test prediction
    sample = [[20,1],[21,0]]  # Example: Iris-setosa
    prediction = predict_music(sample, model_path)
    print(f"Prediction for sample {sample}: {prediction}")

if __name__ == "__main__":
    main()
