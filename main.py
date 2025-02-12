import argparse
import os
from model_pipline import load_data, prepare_data, balance_data, train_model, evaluate_model, save_model, load_model
from sklearn.model_selection import train_test_split

def train(args):
    """Train and save a model."""
    train_path = os.path.join("data", args.train_data)

    if not os.path.exists(train_path):
        print(f"Error: Training data file '{train_path}' not found.")
        return

    # Load and prepare data
    df_train = load_data(train_path)
    df_train_scaled, encoders, scaler = prepare_data(df_train)

    # Split into features and target
    X_train = df_train_scaled.drop(columns=['Churn'])
    y_train = df_train_scaled['Churn']

    # Balance the data
    X_train_res, y_train_res = balance_data(X_train, y_train)

    # Train model
    model = train_model(X_train_res, y_train_res, model_type=args.model)

    # Save model
    save_model(model, args.save_model)
    print(f"Model trained and saved at {args.save_model}")

def test(args):
    """Load a trained model and evaluate it."""
    test_path = os.path.join("data", args.test_data)

    if not os.path.exists(test_path):
        print(f"Error: Testing data file '{test_path}' not found.")
        return

    if not os.path.exists(args.load_model):
        print(f"Error: Model file '{args.load_model}' not found.")
        return

    # Load test data
    df_test = load_data(test_path)
    df_test_scaled, _, _ = prepare_data(df_test)

    # Split into features and target
    X_test = df_test_scaled.drop(columns=['Churn'])
    y_test = df_test_scaled['Churn']

    # Load and evaluate model
    model = load_model(args.load_model)
    report, accuracy = evaluate_model(model, X_test, y_test)

    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:\n", report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Training parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--train_data", type=str, required=True, help="Training dataset filename (inside data/)")
    train_parser.add_argument("--model", type=str, choices=['RF', 'KNN', 'DT'], default='RF', help="Type of model to train")
    train_parser.add_argument("--save_model", type=str, default="models/model.pkl", help="Path to save the trained model")
    train_parser.set_defaults(func=train)

    # Testing parser
    test_parser = subparsers.add_parser("test", help="Evaluate a trained model")
    test_parser.add_argument("--test_data", type=str, required=True, help="Testing dataset filename (inside data/)")
    test_parser.add_argument("--load_model", type=str, required=True, help="Path to load a pre-trained model")
    test_parser.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)
