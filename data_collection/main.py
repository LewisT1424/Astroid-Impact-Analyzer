from data_collection import AstroidAPI
from preprocessing import Preprocessor
from utils import logger, save_df_to_pqt
import time
import asyncio

async def main():
    # Init data collection api
    data_collection_time = time.time()
    api = AstroidAPI()
    start_date = '1925-12-14'
    end_date = '2025-07-08'
    # Make request and data is stored in data file
    await api.make_batch_request(start_date=start_date, end_date=end_date)
    logger.info(f"All data stored in data/raw")


    # Init preprocessor
    preproc = Preprocessor()
    # Call preprocessing pipeline
    final_data = preproc.pipeline()
    # Save data to parquet
    save_df_to_pqt(final_data)
    elapsed_time = time.time() - data_collection_time 
    logger.info(f"Data saved to pqt data/cleaned in {elapsed_time:.2f}s")


if __name__ == '__main__':
    asyncio.run(main())