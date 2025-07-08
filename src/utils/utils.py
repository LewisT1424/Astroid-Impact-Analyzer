import logging
from datetime import datetime, timedelta
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def create_date_range(start_date: str, end_date: str):
    # Format: 2020-01-01
    start_date_1 = datetime.fromisoformat(start_date)
    end_date_1 = datetime.fromisoformat(end_date)

    dates = []
    current_date = start_date_1

    while current_date <= end_date_1:
        dates.append(current_date)
        current_date += timedelta(weeks=1)

    return [date.strftime("%Y-%m-%d") for date in dates]

def save_df_to_pqt(data: pl.DataFrame, train = False):
    if train:
        path = f"data/training_ready/data_{str(datetime.now())[:10]}.parquet"
    else:
        path = f"data/cleaned/data_{str(datetime.now())[:10]}.parquet"
    data.write_parquet(path)
    logger.info(f"Data wrote to: {path}")

def dist_view(data):
    for col in data.columns:
        try: 
            plt.figure(figsize=(10,14))
            sns.kdeplot(data=data, x=col, hue='is_potentially_hazardous')
            plt.show()
        
        except Exception as e:
            raise e
        
def get_data_summary(data):
    # Generate data summary for me to review
    logger.info("=== DATA SUMMARY ===")
    logger.info(f"Shape: {data.shape}")
    logger.info(f"\nColumn types:")
    logger.info(data.dtypes)

    logger.info(f"\nFirst 5 rows:")
    logger.info(data.head())

    logger.info(f"\nBasic statistics:")
    logger.info(data.describe())

    logger.info(f"\nClass distribution:")
    logger.info(y.value_counts())

    # Check for any obvious scaling issues
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    logger.info(f"\nNumeric columns ranges:")
    for col in numeric_cols[:10]:  # First 10 numeric columns
        logger.info(f"{col}: {data[col].min():.3f} to {data[col].max():.3f}")

def save_model(model):
    try:
        path = 'models/astroid_xgboost_model.pkl'
        with open(path, 'wb') as file:
            pickle.dump(model, file)
            logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            logger.info(f"Model successfully loaded")
            return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")