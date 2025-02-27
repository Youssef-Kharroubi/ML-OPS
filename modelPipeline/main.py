import argparse
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_pipline import load_model, prepare_data

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
# Global variables
model = None
encoders = None
scaler = None

def preprocess_input(data, encoders, scaler):
    """Preprocess input data using saved encoders and scaler"""
    df = pd.DataFrame([data])
    
    # Apply label encoding to categorical variables
    for column, encoder in encoders.items():
        if column in df.columns:
            try:
                df[column] = encoder.transform(df[column])
            except ValueError as e:
                raise ValueError(f"Invalid value in column {column}: {str(e)}")
    
    # Apply scaling
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df_scaled

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required features
        required_features = [
            "State", "Account length", "Area code", "International plan",
            "Voice mail plan", "Number vmail messages", "Total day minutes",
            "Total day calls", "Total day charge", "Total eve minutes",
            "Total eve calls", "Total eve charge", "Total night minutes",
            "Total night calls", "Total night charge", "Total intl minutes",
            "Total intl calls", "Total intl charge", "Customer service calls"
        ]
        
        for feature in required_features:
            if feature not in data:
                return jsonify({
                    "error": f"Missing required feature: {feature}"
                }), 400
        
        # Preprocess the input data
        try:
            processed_data = preprocess_input(data, encoders, scaler)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        # Prepare response
        response = {
            "prediction": int(prediction[0]),  # 0: No churn, 1: Churn
            "churn_probability": float(prediction_proba[0][1]),
            "status": "success"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def serve(args):
    global model, encoders, scaler
    if not os.path.exists(args.load_model):
        print(f"❌ Error: Model file '{args.load_model}' not found.")
        return
    try:
        model_data = load_model(args.load_model)
        model, encoders, scaler = model_data
        print(f"✨ Model loaded successfully")
        print(f"🌐 Starting server at http://localhost:{args.port}")
        app.run(host='0.0.0.0', port=args.port)
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a ML Model")
    parser.add_argument("--load_model", type=str, required=True, help="Path to load a pre-trained model")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()
    serve(args)