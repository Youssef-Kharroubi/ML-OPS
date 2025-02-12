import argparse
import os
from model_pipline import load_data, prepare_data, balance_data, train_model, evaluate_model, save_model, load_model
from sklearn.model_selection import train_test_split

def main(args):
    train_path = os.path.join("data", args.train_data)
    test_path = os.path.join("data", args.test_data)

    # Check if training and testing data files exist
    if not os.path.exists(train_path):
        print(f"Error: Training data file '{train_path}' not found.")
        return

    if not os.path.exists(test_path):
        print(f"Error: Testing data file '{test_path}' not found.")
        return

    # Load data
    df_train = load_data(train_path)
    df_test = load_data(test_path)


    # Split into features and target
    X_train = df_train.drop(columns=['Churn'])
    y_train = df_train['Churn']
    X_test = df_test.drop(columns=['Churn'])
    y_test = df_test['Churn']

    # Prepare data (scaling and encoding)
    X_train, y_train, encoders, scaler = prepare_data(df_train, target_column='Churn')
    X_test, y_test, _, _ = prepare_data(df_test, target_column='Churn')

    # Balance the data (handle class imbalance)
    X_train_res, y_train_res = balance_data(X_train, y_train.astype(int))  # Ensure y_train is categorical

    
    if args.load_model and os.path.exists(args.load_model):
        model = load_model(args.load_model)
    else:
        model = train_model(X_train_res, y_train_res, model_type=args.model)
        save_model(model, args.save_model)
    
    report, accuracy = evaluate_model(model, X_test, y_test)
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Arguments for data files
    parser.add_argument("--train_data", type=str, required=True, help="Training dataset filename (inside data/)")
    parser.add_argument("--test_data", type=str, required=True, help="Testing dataset filename (inside data/)")

    # Model choice
    parser.add_argument("--model", type=str, choices=['RF', 'KNN', 'DT'], default='RF', help="Type of model to train")

    # Path to save the trained model
    parser.add_argument("--save_model", type=str, default="models/model.pkl", help="Path to save the trained model")

    # Path to load an existing model
    parser.add_argument("--load_model", type=str, help="Path to load a pre-trained model")

    args = parser.parse_args()

    # Call main function
    main(args)

