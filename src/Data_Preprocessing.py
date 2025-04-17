from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Data Preprocessing')
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'Data_Preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

def preprocess_df(df: pd.DataFrame, target_column = 'HeartDisease', binary_columns = ['Gender']) -> pd.DataFrame:
    """Preprocesses the Training and Testing Data by Standard Scaler"""
    try:
        logger.debug('Preprocessing Data starts....')
        X = df.drop(columns = target_column)
        y = df[target_column]

        cols_to_keep = [col for col in X.columns if col not in binary_columns]
        cols_to_scale = binary_columns

        scaler = StandardScaler()
        X_scaler_array = scaler.fit_transform(X[cols_to_scale])
        X_scaled = pd.DataFrame(X_scaler_array, columns=cols_to_scale, index=df.index)

        X_unscaled = X[cols_to_keep]
        X_final = pd.concat([X_scaled, X_unscaled], axis = 1)
        df_processed = pd.concat([X_final, y], axis = 1)

        logger.debug('Preprocessing Data ends....')
        return df_processed
    except Exception as e:
        logger.error('Unexpected error: {}'.format(e))
        raise

def main():
    """Main function to load raw data and preprocess it and save the processed data"""
    try:
        train_data = pd.read_csv('./artifacts/raw/train.csv')
        test_data = pd.read_csv('./artifacts/raw/test.csv')
        logger.debug('Raw data loaded...')

        processed_train_data = preprocess_df(train_data)
        processed_test_data = preprocess_df(test_data)
        logger.debug('Processed data gathered...')

        data_path = os.path.join('./artifacts', 'interim')
        os.makedirs(data_path, exist_ok=True)

        processed_train_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        processed_test_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.debug('Processed data saved to {}'.format(data_path))
    except FileNotFoundError as e:
        logger.error('Raw data not found.')
    except pd.errors.EmptyDataError as e:
        logger.error('Raw data empty.')
    except Exception as e:
        logger.error('Unexpected error during the Data Preprocessing Process: {}'.format(e))

if __name__ == '__main__':
    main()