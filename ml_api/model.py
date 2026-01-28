import joblib
import os
from sklearn.linear_model import LinearRegression

MODEL_PATH="models/linear_regression.pkl"

def create_model():
    model = LinearRegression()
    return model

def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)