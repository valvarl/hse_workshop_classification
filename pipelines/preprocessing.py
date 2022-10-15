import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os

import src.config as cfg
from src import utils

import sys
print(sys.path)

raw_path = 'data/raw/'
print(os.path.abspath(os.path.join(raw_path, 'train.csv')))

train = pd.read_csv(os.path.join(raw_path, 'train.csv'))
test = pd.read_csv(os.path.join(raw_path, 'test.csv'))

train, target = train.drop(cfg.TARGET_COLS, axis=1), train[cfg.TARGET_COLS]

def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df

def drop_unnecesary_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'ID_y' in df.columns:
        df = df.drop('ID_y', axis=1)
    return df

def add_ord_edu(df: pd.DataFrame) -> pd.DataFrame:
    df[f'{cfg.EDU_COL}_ord'] = df[cfg.EDU_COL].str.slice(0, 1).astype(np.int8).values
    return df

def fill_sex(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[cfg.SEX_COL].value_counts().index[0]
    df[cfg.SEX_COL] = df[cfg.SEX_COL].fillna(most_freq)
    return df

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')

    ohe_int_cols = train[cfg.OHE_COLS].select_dtypes('number').columns
    df[ohe_int_cols] = df[ohe_int_cols].astype(np.int8)

    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, cfg.ID_COL)
    df = drop_unnecesary_id(df)
    df = add_ord_edu(df)
    df = fill_sex(df)
    df = cast_types(df)
    return df

train = train.pipe(preprocess)
test = test.pipe(preprocess)

processed_data_path = 'data/processed/'
if not os.path.exists(processed_data_path): 
    os.makedirs(processed_data_path) 
utils.save_as_pickle(train, os.path.join(processed_data_path, 'train.pkl'))
utils.save_as_pickle(target, os.path.join(processed_data_path, 'target.pkl'))
utils.save_as_pickle(test, os.path.join(processed_data_path, 'test.pkl'))

