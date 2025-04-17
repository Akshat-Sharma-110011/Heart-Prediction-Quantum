import os
import logging
import pandas as pd
import numpy as np

# ---- Setup Logging ----
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

# ---- Feature Engineering Function ----
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Feature Engineering Columns"""
    try:
        df_fe = df.copy()

        logger.debug('Feature engineering starts...')

        # ---- Check required columns ----
        required_columns = ['HeartRate', 'Age', 'BloodPressure', 'Cholesterol', 'QuantumPatternFeature']
        missing = [col for col in required_columns if col not in df_fe.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        # ---- Heart Rate Based Features ----
        df_fe['IsTachycardic'] = (df_fe['HeartRate'] > 100).astype(int)
        df_fe['IsBradycardic'] = (df_fe['HeartRate'] < 60).astype(int)
        df_fe['PercentMaxHR'] = df_fe['HeartRate'] / np.clip((220 - df_fe['Age']), 1e-5, None)
        df_fe['HR_per_BP'] = df_fe['HeartRate'] / (df_fe['BloodPressure'] + 1e-5)
        logger.debug('Heart Rate Based Features engineered.')

        # ---- QuantumPatternFeature Transformations ----
        df_fe['QPF_squared'] = df_fe['QuantumPatternFeature'] ** 2
        df_fe['QPF_log'] = np.where(df_fe['QuantumPatternFeature'] > -1,
                                    np.log1p(df_fe['QuantumPatternFeature']), np.nan)
        logger.debug('Quantum Pattern Feature Transformations done.')

        # ---- Interaction Features ----
        df_fe['Age_Cholesterol'] = df_fe['Age'] * df_fe['Cholesterol']
        df_fe['Age_BP'] = df_fe['Age'] * df_fe['BloodPressure']
        df_fe['BP_Cholesterol'] = df_fe['BloodPressure'] * df_fe['Cholesterol']
        df_fe['Cholesterol_minus_BP'] = df_fe['Cholesterol'] - df_fe['BloodPressure']
        df_fe['HR_Cholesterol'] = df_fe['HeartRate'] * df_fe['Cholesterol']
        logger.debug('Interaction Features engineered.')

        # ---- Clinical Risk Proxy Score ----
        df_fe['ClinicalRiskScore'] = (
            0.02 * df_fe['Age'] +
            0.03 * df_fe['BloodPressure'] +
            0.04 * df_fe['Cholesterol'] +
            0.05 * df_fe['HeartRate']
        )
        logger.debug('Clinical Risk Proxy Score calculated.')

        # ---- One-Hot Encoding ----
        cat_cols = df_fe.select_dtypes(include=['category', 'object']).columns
        df_fe = pd.get_dummies(df_fe, columns=cat_cols, drop_first=True)

        # ---- Convert Binary Columns to Int ----
        for col in df_fe.columns:
            if df_fe[col].nunique() == 2 and df_fe[col].dtype != 'int':
                df_fe[col] = df_fe[col].fillna(0).astype(int)

        df_fe['QPF_log'].fillna(df_fe['QPF_log'].mean(), inplace=True)

        logger.debug('One-hot encoding and type conversion done.')

        return df_fe

    except KeyError as e:
        logger.error(f'Missing columns in input data: {e}')
        raise
    except ValueError as e:
        logger.error(f'Value error during transformations: {e}')
        raise
    except Exception as e:
        logger.error(f'Unknown error during transformations: {e}')
        raise

# ---- Load Data Function ----
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug(f'Data loaded from {data_url}')
        return df
    except FileNotFoundError:
        logger.error(f'File not found: {data_url}')
        raise
    except pd.errors.ParserError:
        logger.error(f'Parsing error while loading data from: {data_url}')
        raise
    except UnicodeDecodeError:
        logger.error(f'Encoding issue while reading file: {data_url}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error while loading data: {e}')
        raise

# ---- Save Data Function ----
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save dataframe to a CSV file"""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f'Data saved to {file_path}')
    except FileExistsError:
        logger.error(f'File already exists: {file_path}')
        raise
    except Exception as e:
        logger.error(f'Error during saving data: {e}')
        raise

# ---- Main Execution ----
def main():
    try:
        train_data = load_data('./artifacts/interim/train_processed.csv')
        test_data = load_data('./artifacts/interim/test_processed.csv')

        train_engineered = feature_engineer(train_data)
        test_engineered = feature_engineer(test_data)

        save_data(train_engineered, './artifacts/processed/train_engineered.csv')
        save_data(test_engineered, './artifacts/processed/test_engineered.csv')

    except Exception as e:
        logger.error(f'Unexpected error during pipeline execution: {e}')

if __name__ == '__main__':
    main()