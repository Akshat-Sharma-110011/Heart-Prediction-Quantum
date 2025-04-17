import os
import numpy as np
import pandas as pd
import pickle
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ========== Logging Setup ==========
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Model Training')
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'Model_Training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

# ========== Core Functions ==========

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug(f'Data loaded from {data_url} with shape {df.shape}')
        return df
    except Exception as e:
        logger.error(f'Error while loading data: {e}')
        raise

def validate_data(df: pd.DataFrame, target_column: str):
    """Validate if the dataset has correct structure."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values detected in the dataset.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError("Target column must be numeric (0/1 for classification).")

def train_model(X_train: pd.DataFrame, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    try:
        logger.debug('Initializing Random Forest with params: {}'.format(params))
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        logger.debug('Model training completed.')
        return clf
    except Exception as e:
        logger.error(f'Error while training model: {e}')
        raise

def evaluate_model(model, X_val, y_val):
    """Evaluate model and log accuracy"""
    try:
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        logger.debug(f"Classification Report:\n{report}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

def save_model(model, file_path: str):
    """Save the trained model to disk."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f'Model saved at: {file_path}')
    except Exception as e:
        logger.error(f'Error saving model: {e}')
        raise

# ========== Main Pipeline ==========

def main():
    try:
        # Step 1: Load Data
        data_path = './artifacts/processed/train_engineered.csv'
        df = load_data(data_path)

        # Step 2: Validate Data
        target_column = 'HeartDisease'
        validate_data(df, target_column)

        # Step 3: Split Features/Target
        X = df.drop(columns=target_column)
        y = df[target_column]

        logger.debug(f"Features shape: {X.shape} | Target distribution:\n{y.value_counts()}")

        # Step 4: Split train/test for evaluation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Step 5: Define model parameters
        params = {'n_estimators': 79, 'max_depth': 3, 'random_state': 42}

        # Step 6: Train Model
        model = train_model(X_train, y_train, params)

        # Step 7: Evaluate Model
        evaluate_model(model, X_val, y_val)

        # Step 8: Save Model
        model_save_path = 'models/model.pkl'
        save_model(model, model_save_path)

    except Exception as e:
        logger.error(f"Fatal error in training pipeline: {e}")

if __name__ == '__main__':
    main()
