import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
# from nlp.news_extractor import SectorNewsExtractor, sector_keywords
# from nlp.word_cloud import SectorWordCloud

def drawdown(s):
    running_max = s.cummax()
    dd = (running_max - s) / running_max
    return dd

def get_model_dict() -> dict:
    path = "Anomalies Dataset"
    model_list = sorted([x.split('.')[0] for x in next(os.walk(path))[2]])
    model_dict = {}
    for model in model_list:
        data = pd.read_csv(f"{path}/{model}.csv", index_col='date')
        data.index = pd.to_datetime(data.index)
        model_dict[model] = data
    return model_dict

def get_ohlc_dict():
    path = "Data"
    df_list = sorted([x.split('.')[0] for x in next(os.walk(path))[2]])
    df_dict = {}
    for df in df_list:
        data = pd.read_csv(f"{path}/{df}.csv", index_col='date')
        data.index = pd.to_datetime(data.index)
        df_dict[df] = data
    return df_dict

sectors = ['Materials', 'Industrials', 'Health Care', 'Real Estate', 'Consumer Discretionary', 'Financials', 
               'Utilities', 'Information Technology', 'Energy', 'Consumer Staples', 'Communication Services']
def load_sector_anomalies(data_path, sector):
    model_list = sorted([x.split('.')[0] for x in next(os.walk(data_path))[2]])
    tmp = []
    for model in model_list:
        model_data = pd.read_csv(f"{data_path}/{model}.csv", index_col='date')
        model_data.index = pd.to_datetime(model_data.index)
        sector_anomalies = (model_data.loc[:, sector]).astype(int).rename(model)
        tmp.append(sector_anomalies)
    output = pd.concat(tmp, axis=1)
    return output

def load_sector_return(data_path, sector):
    log_return = pd.read_csv(f"{data_path}/log_ret.csv", index_col='date')
    log_return.index = pd.to_datetime(log_return.index)
    output = log_return.loc[:, sector]
    return output


def load_sector_drawdown(data_path, sector):
    log_return = pd.read_csv(f"{data_path}/log_ret.csv", index_col='date')
    log_return.index = pd.to_datetime(log_return.index)
    dd = drawdown(np.exp(log_return.loc[:, sector].cumsum()))
    return dd

        





