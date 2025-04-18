import os
import numpy as np
import pandas as pd
import pickle
import logging
import yaml

from sklearn.ensemble import RandomForestClassifier

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
    logger.debug(f'Data validation passed for target column: {target_column}')

def train_model(X_train: pd.DataFrame, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    try:
        logger.debug(f'Initializing Random Forest with params: {params}')
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        logger.debug('Model training completed.')
        return clf
    except Exception as e:
        logger.error(f'Error while training model: {e}')
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

def load_params(params_path: str) -> dict:
    """Load model parameters from a YAML file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
            logger.debug(f'Parameters loaded from {params_path}')
        return params
    except FileNotFoundError as e:
        logger.error(f'Parameter file not found: {e}')
        raise
    except yaml.YAMLError as e:
        logger.error(f'YAML parsing error: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error loading parameters: {e}')
        raise

# ========== Main Training Pipeline ==========

def main():
    try:
        target_col = 'HeartDisease'
        model_save_path = 'models/model.pkl'
        train_path = './artifacts/processed/train_engineered.csv'
        params_path = 'params.yaml'

        # Load and validate data
        train_data = load_data(train_path)
        validate_data(train_data, target_col)

        # Load training parameters
        params = load_params(params_path)['Model_Training']

        # Split features and target
        X_train = train_data.drop(columns=target_col)
        y_train = train_data[target_col]

        # Train and save model
        model = train_model(X_train, y_train, params)
        save_model(model, model_save_path)

        logger.info('Model training pipeline completed successfully.')

    except Exception as e:
        logger.error(f"Fatal error in training pipeline: {e}")

if __name__ == '__main__':
    main()
