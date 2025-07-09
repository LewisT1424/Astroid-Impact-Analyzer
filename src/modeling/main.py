from model_creation import Model
import polars as pl
from src.utils.utils import save_model

def main():
    data = pl.read_parquet('data/training_ready/data_2025-07-08.parquet')
    model = Model(data=data).run_model_creation_pipeline()
    # Save model
    save_model(model)
    return model

if __name__ == '__main__':
    main()