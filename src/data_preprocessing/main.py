import polars as pl
from src.data_preprocessing.transformations import Preprocessor
from datetime import datetime
from src.utils.utils import save_df_to_pqt, logger

def main():
    preprocessor = Preprocessor()
    data, feature_names, scaler = preprocessor.run_pipeline()

    save_df_to_pqt(data, train=True)

if __name__ == '__main__':
    main()

