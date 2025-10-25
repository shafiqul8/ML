

# src/model_trainer.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

def train_model(df):
    X = df.drop(columns=["genre"])
    y = df["genre"]

    # Split into train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n📊 Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "music_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved at: {model_path}")

    return model_path
