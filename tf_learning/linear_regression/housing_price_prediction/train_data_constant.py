# -*- coding: utf-8 -*-
import os

import pandas as pd

HOUSE_PRICES_DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "hourse_price_datas")
HOUSE_PRICES_TRAIN_DATA = os.path.join(HOUSE_PRICES_DATA_DIR, "house_prices_train.csv")
HOUSE_PRICES_PREDICT_DATA = os.path.join(HOUSE_PRICES_DATA_DIR, "house_prices_predict.csv")


def read_csv_data(data_path, setp=","):
    return pd.read_csv(data_path, sep=setp)


def read_house_train_data():
    return read_csv_data(HOUSE_PRICES_TRAIN_DATA)


def read_house_predict_data():
    return read_csv_data(HOUSE_PRICES_PREDICT_DATA)
