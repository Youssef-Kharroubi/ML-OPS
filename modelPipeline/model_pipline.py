import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def prepare_data(df, target_column='Churn'):
    """Perform data cleaning, encoding, and scaling."""
    label_encoders = {}
    # Encode categorical variables except target
    for column in df.select_dtypes(include=['object']).columns:
        if column != target_column:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Separate features and target before scaling
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Standardize features (not target)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y.astype(int), label_encoders, scaler  # Convert y to int

def balance_data(X, y, method='SMOTE'):
    """Balance dataset using oversampling or undersampling."""
    if method == 'SMOTE':
        sampler = SMOTE()
    elif method == 'ADASYN':
        sampler = ADASYN()
    elif method == 'RandomUnderSampler':
        sampler = RandomUnderSampler()
    else:
        raise ValueError("Invalid resampling method")

    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

def train_model(X_train, y_train, model_type='RF'):
    """Train a machine learning model."""
    if model_type == 'RF':
        model = RandomForestClassifier()
    elif model_type == 'KNN':
        model = KNeighborsClassifier()
    elif model_type == 'DT':
        model = DecisionTreeClassifier()
    else:
        raise ValueError("Invalid model type")
    
    model.fit(X_train, y_train)  # Fit first, log later in run_pipeline.py
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return report, accuracy

def save_model(model, filename):
    """Save trained model using joblib."""
    joblib.dump(model, filename)

def load_model(filename):
    """Load saved model using joblib."""
    return joblib.load(filename)