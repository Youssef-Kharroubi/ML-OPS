from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load the model
MODEL_PATH = os.getenv("MODEL_PATH", "/models/churn_model.pkl")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Model not found at {MODEL_PATH}")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])  # Convert input to DataFrame
        
        prediction = model.predict(df)[0]  # Make prediction
        result = {"churn": int(prediction)}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
