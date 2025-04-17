import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
import pandas as pd
import numpy as np
import logging
import pickle
import json

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Feature Engineering')
logger.setLevel(logging.DEBUG)

# Avoid adding duplicate handlers
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    log_file_path = os.path.join(log_dir, 'Feature_Engineering.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

def load_model(file_path: str):
    """Load Model from the Models folder"""
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug("Model loaded from {}".format(file_path))
        return model
    except FileNotFoundError:
        logger.error("Model could not be loaded from {}".format(file_path))
        raise
    except Exception as e:
        logger.error(f'Unexpected error while evaluating model {e}')
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug(f'Data loaded from {data_url} with shape {df.shape}')
        return df
    except Exception as e:
        logger.error(f'Error while loading data: {e}')
        raise

def evaluate_model(clf, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Evaluate model on test data."""
    try:
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        roc_auc = roc_auc_score(y_test, y_score)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
        }
        logger.debug('Model evaluation complete, Metrics formed')
        return metrics_dict
    except Exception as e:
        logger.error('Unexpected error while evaluating model: {}'.format(e))
        raise

def save_metrics(metrics_dict: dict, file_path: str) -> None:
    """Save metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics_dict, f)
        logger.debug('Metrics saved to {}'.format(file_path))
    except Exception as e:
        logger.error('Unexpected error while saving metrics: {}'.format(e))
        raise

def main():
    try:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./artifacts/processed/test_engineered.csv')

        X_test = test_data.drop(columns=['HeartDisease'])
        y_test = test_data['HeartDisease']

        metrics_dict = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics_dict, 'reports/metrics.json')
    except Exception as e:
        logger.error('Unexpected error while evaluating model: {}'.format(e))
        raise

if __name__ == '__main__':
    main()