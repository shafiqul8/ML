

# src/utils.py
import pandas as pd
from sklearn.datasets import load_iris

def load_iris_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df
