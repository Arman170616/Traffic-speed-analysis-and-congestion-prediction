# src/utils.py
import joblib
import os

def save_model(model, filepath='model.joblib'):
    """Save the model to a file."""
    joblib.dump(model, filepath)

def load_model(filepath='model.joblib'):
    """Load the model from a file."""
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        print("Model file does not exist.")
        return None
