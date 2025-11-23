import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import preprocessing functions from preprocess.py
from src.preprocess import load_data, clean_data, remove_outliers, scale_features, split_data

def train_model():
    # Load and preprocess data
    df = load_data()
    df = clean_data(df)
    df = remove_outliers(df)
    df, scaler = scale_features(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Define the model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=8, 
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)
    
    # Validation predictions
    val_pred = model.predict(X_val)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_val, val_pred)
    precision = precision_score(y_val, val_pred)
    recall = recall_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)

    print("\n===== Model Performance (Validation Set) =====")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    # Save model and scaler
    joblib.dump(model, "models/anomaly_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nModel and scaler saved successfully in /models folder!")

if __name__ == "__main__":
    train_model()
