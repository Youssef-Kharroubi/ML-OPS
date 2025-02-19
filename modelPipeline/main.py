import argparse
import os
import pandas as pd
from model_pipline import (
    load_data, prepare_data, balance_data, train_model, evaluate_model, save_model, load_model
)


def train(args):
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
    parser = argparse.ArgumentParser(description="Train, Evaluate, or Predict using a ML Model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--train_data", type=str, required=True, help="Training dataset filename (inside data/)")
    train_parser.add_argument("--model", type=str, choices=['RF', 'KNN', 'DT'], default='RF', help="Type of model to train")
    train_parser.add_argument("--save_model", type=str, default="models/model.pkl", help="Path to save the trained model")
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("test", help="Evaluate a trained model")
    eval_parser.add_argument("--test_data", type=str, required=True, help="Testing dataset filename (inside data/)")
    eval_parser.add_argument("--load_model", type=str, required=True, help="Path to load a pre-trained model")
    eval_parser.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)
