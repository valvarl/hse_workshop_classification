# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in df.dtypes.items():
        if dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif dtype == np.int64:
            df[col] = df[col].astype(np.int32)
    return df


def save_as_pickle(df: pd.DataFrame, path: str) -> None:
    df.to_pickle(path)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    df = pd.read_csv(input_filepath)
    df = cast_types(df)
    save_as_pickle(df, output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
