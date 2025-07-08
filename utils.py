import logging
from datetime import datetime, timedelta
import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def save_df_to_pqt(data: pl.DataFrame):
    path = f"data/cleaned/data_{str(datetime.now())[:10]}.parquet"
    data.write_parquet(path)
    logger.info(f"Data wrote to: {path}")