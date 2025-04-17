import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Data Ingestion')
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'Data_Ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug(f'Data loaded from {data_url}')
        return df
    except pd.errors.ParserError:
        logger.error(f'Could not load data from {data_url}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error while loading data from {data_url}')
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data"""
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.debug(f'Data Preprocessed using dropping NA and dropping duplicates')
        return df
    except Exception as e:
        logger.error(f'Unexpected error while preprocessing data from {df}')
        raise

def save_data(train_data_df: pd.DataFrame, test_data_df: pd.DataFrame, data_path: str) -> None:
    """Save the training and test data"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data_df.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data_df.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug(f'Data saved to {data_path}')
    except Exception as e:
        logger.error(f'Unexpected error while saving data from {data_path}')
        raise

def main():
    try:
        test_size = 0.2
        data_path = 'D:\\MLOps Projects\\Heart-Prediction-Quantum\\experiments\\Heart Prediction Quantum Dataset.csv'
        save_dir = 'artifacts'
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_data_df, test_data_df = train_test_split(final_df, test_size=test_size)
        save_data(train_data_df, test_data_df, save_dir)
    except Exception as e:
        logger.error('Failed to complete the Data Ingestion process')
        print(f'Error: {e}')


if __name__ == '__main__':
    main()