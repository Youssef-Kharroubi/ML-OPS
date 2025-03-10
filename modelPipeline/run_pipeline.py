import os
import subprocess
import argparse
import pandas as pd
import numpy as np
import mlflow
import psutil
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix
)
from model_pipline import (
    load_data, prepare_data, balance_data, train_model, evaluate_model, save_model , load_model
)
from elasticsearch import Elasticsearch
from datetime import datetime

es = Elasticsearch("http://elasticsearch:9200")

def plot_roc_curve(y_true, y_pred_proba, title, filename):
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()
    return auc_score

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(filename)
    plt.close()

def get_system_metrics():
    """Collect system metrics using psutil."""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    metrics = {
        "cpu_usage_percent": cpu_usage,
        "memory_used_mb": memory.used / 1024 / 1024,
        "memory_total_mb": memory.total / 1024 / 1024,
        "disk_used_gb": disk.used / 1024 / 1024 / 1024,
        "disk_total_gb": disk.total / 1024 / 1024 / 1024
    }
    return metrics

def check_model_performance(args):
    test_path = os.path.join("data", args.test_data)
    model_path = args.load_model

    if not os.path.exists(model_path):
        print(f"No existing model at {model_path}. Triggering training.")
        return False

    if not os.path.exists(test_path):
        print(f"Error: Test data not found at {test_path}")
        exit(1)

    model_data = load_model(model_path)
    model, _, _ = model_data
    df_test = load_data(test_path)
    X_test, y_test, _, _ = prepare_data(df_test)

    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)

    ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", 0.80))
    ROC_AUC_THRESHOLD = float(os.getenv("ROC_AUC_THRESHOLD", 0.80))

    print(f"Current Test Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    if accuracy < ACCURACY_THRESHOLD or roc_auc < ROC_AUC_THRESHOLD:
        print(f"Performance below thresholds (Accuracy: {ACCURACY_THRESHOLD}, ROC-AUC: {ROC_AUC_THRESHOLD}). Triggering retraining.")
        return False
    else:
        print("Performance meets benchmarks. Using existing model.")
        return True
    
def train(args):
    """Train the model and log to MLflow."""
    train_path = os.path.join("data", args.train_data)
    model_path = os.path.abspath(args.save_model)
    print(f"🔍 Training with data at: {train_path}")

    df_train = load_data(train_path)
    X_train, y_train, encoders, scaler = prepare_data(df_train)
    X_train_res, y_train_res = balance_data(X_train, y_train)
    mlflow.set_tracking_uri("http://mlflow-ui:5001")
    mlflow.set_experiment("Churn_Prediction_Experiment")
    with mlflow.start_run():
        mlflow.log_param("model", args.model)
        model = train_model(X_train_res, y_train_res, model_type=args.model)

        # Training metrics
        y_train_pred = model.predict(X_train_res)
        y_train_pred_proba = model.predict_proba(X_train_res)[:, 1]  # Probability of positive class
        train_accuracy = accuracy_score(y_train_res, y_train_pred)
        train_precision = precision_score(y_train_res, y_train_pred)
        train_recall = recall_score(y_train_res, y_train_pred)
        train_f1 = f1_score(y_train_res, y_train_pred)

        print(f"Train Accuracy: {train_accuracy:.4f}")
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f1", train_f1)

        # ROC-AUC for training
        train_roc_auc = plot_roc_curve(y_train_res, y_train_pred_proba, "ROC Curve - Training", "train_roc_curve.png")
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        mlflow.log_artifact("train_roc_curve.png")

        # Confusion Matrix for training
        plot_confusion_matrix(y_train_res, y_train_pred, "Confusion Matrix - Training", "train_confusion_matrix.png")
        mlflow.log_artifact("train_confusion_matrix.png")

        # Log model with input example
        input_example = X_train_res.iloc[:1]

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name="Churn_Prediction_Model"
        )
        print("✅ Model trained and logged to MLflow")
        save_model((model, encoders, scaler), model_path)

def test(args):
    """Test the model and log to MLflow."""
    test_path = os.path.join("data", args.test_data)
    model_path = args.load_model
    if not os.path.exists(test_path) or not os.path.exists(model_path):
        print(f"Error: File not found - Test data: {test_path}, Model: {model_path}")
        exit(1)

    model_data = load_model(model_path)
    model, _, _ = model_data
    df_test = load_data(test_path)
    X_test, y_test, _, _ = prepare_data(df_test)
    mlflow.set_tracking_uri("http://mlflow-ui:5001")


    mlflow.set_experiment("Churn_Prediction_Experiment")
    with mlflow.start_run(run_name="Model_Test_Metrics"):
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        print(f"\n🔹 Model Evaluation Results (Test):")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)

        # ROC-AUC for test
        test_roc_auc = plot_roc_curve(y_test, y_test_pred_proba, "ROC Curve - Test", "test_roc_curve.png")
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_artifact("test_roc_curve.png")

        # Confusion Matrix for test
        plot_confusion_matrix(y_test, y_test_pred, "Confusion Matrix - Test", "test_confusion_matrix.png")
        mlflow.log_artifact("test_confusion_matrix.png")
        es.index(index="ml_metrics", body={
                "@timestamp": datetime.utcnow().isoformat(),
                "stage": "test",
                "model": args.model,
                "accuracy": test_accuracy,
                "precision": test_precision,
                "recall": test_recall,
                "f1_score": test_f1,
                "roc_auc": test_roc_auc
            })

 
    mlflow.set_experiment("System_Metrics_Experiment")
    with mlflow.start_run(run_name="System_Metrics"):
        sys_metrics = get_system_metrics()
        mlflow.log_metrics(sys_metrics)
        sys_metrics["@timestamp"] = datetime.utcnow().isoformat()
        es.index(index="system_metrics", body=sys_metrics)

if __name__ == "__main__":
    PYTHON = os.getenv("PYTHON", "python3")
    MODEL = os.getenv("MODEL", "RF")

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)  

    parser = argparse.ArgumentParser(description="Run ML Pipeline")
    parser.add_argument("--train_data", type=str, default="churn-bigml-80.csv", help="Training dataset filename")
    parser.add_argument("--test_data", type=str, default="churn-bigml-20.csv", help="Testing dataset filename")
    parser.add_argument("--model", type=str, choices=['RF', 'KNN', 'DT'], default=MODEL, help="Model type")
    parser.add_argument("--save_model", type=str, default=f"models/{MODEL}.pkl", help="Path to save the model")
    parser.add_argument("--load_model", type=str, default=f"models/{MODEL}.pkl", help="Path to load the model")
    args = parser.parse_args()

    if not check_model_performance(args):
        train(args)
    test(args)
