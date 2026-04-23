from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler with error handling
def load_model_and_scaler():
    try:
        model = joblib.load("models/model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except (FileNotFoundError, EOFError):
        # Return None during testing or if files don't exist
        return None, None

model, scaler = load_model_and_scaler()

@app.route("/")
def home():
    return "ML Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json["features"]  # Expect list
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)
    prediction = model.predict(data)
    
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
