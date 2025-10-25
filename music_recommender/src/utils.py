

# src/utils.py
import pandas as pd

def load_iris_dataset():
    df = pd.read_csv("music.csv")
    return df
