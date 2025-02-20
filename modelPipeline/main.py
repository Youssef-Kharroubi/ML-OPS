import argparse
import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

from model_pipline import (
    load_data, prepare_data, balance_data, train_model, evaluate_model, save_model, load_model
)

# Initialize the Flask app
app = Flask(__name__)

# Global variables for model, encoders, and scaler
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
    """Initialize and run the Flask server"""
    global model, encoders, scaler
    
    if not os.path.exists(args.load_model):
        print(f"❌ Error: Model file '{args.load_model}' not found.")
        return

    # Load model and assign to global variables
    try:
        model_data = load_model(args.load_model)
        model, encoders, scaler = model_data
        print(f"✨ Model loaded successfully")
        print(f"🌐 Starting server at http://localhost:{args.port}")
        app.run(host='0.0.0.0', port=args.port)
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")
        return

def train(args):
    """Train the model"""
    train_path = os.path.join("data", args.train_data)
    
    if not os.path.exists(train_path):
        print(f"Error: Training data file '{train_path}' not found.")
        return

    model_path = os.path.abspath(args.save_model)
    print(f"🔍 Checking if model exists at: {model_path}")

    if os.path.exists(model_path):
        print("⚠️ Model file already exists.")
        overwrite = input("Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("❌ Training aborted. Model not overwritten.")
            return

    df_train = load_data(train_path)
    X_train, y_train, encoders, scaler = prepare_data(df_train)
    X_train_res, y_train_res = balance_data(X_train, y_train)

    model = train_model(X_train_res, y_train_res, model_type=args.model)
    save_model((model, encoders, scaler), model_path)
    print(f"✅ Model trained and saved at: {model_path}")

def test(args):
    """Test the model"""
    test_path = os.path.join("data", args.test_data)
    if not os.path.exists(test_path):
        print(f"Error: Testing data file '{test_path}' not found.")
        return

    if not os.path.exists(args.load_model):
        print(f"Error: Model file '{args.load_model}' not found.")
        return

    model_data = load_model(args.load_model)
    model, encoders, scaler = model_data

    df_test = load_data(test_path)
    X_test, y_test, _, _ = prepare_data(df_test)

    report, accuracy = evaluate_model(model, X_test, y_test)

    print("\n🔹 Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Evaluate, or Serve a ML Model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train parser
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--train_data", type=str, required=True, help="Training dataset filename (inside data/)")
    train_parser.add_argument("--model", type=str, choices=['RF', 'KNN', 'DT'], default='RF', help="Type of model to train")
    train_parser.add_argument("--save_model", type=str, default="models/model.pkl", help="Path to save the trained model")
    train_parser.set_defaults(func=train)

    # Test parser
    eval_parser = subparsers.add_parser("test", help="Evaluate a trained model")
    eval_parser.add_argument("--test_data", type=str, required=True, help="Testing dataset filename (inside data/)")
    eval_parser.add_argument("--load_model", type=str, required=True, help="Path to load a pre-trained model")
    eval_parser.set_defaults(func=test)

    # Serve parser
    serve_parser = subparsers.add_parser("serve", help="Serve the trained model")
    serve_parser.add_argument("--load_model", type=str, required=True, help="Path to load a pre-trained model")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to run the server on (default: 8080)")
    serve_parser.set_defaults(func=serve)

    # Parse the arguments and run the appropriate function
    args = parser.parse_args()
    args.func(args)