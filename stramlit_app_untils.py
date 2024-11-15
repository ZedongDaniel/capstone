import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
# from nlp.news_extractor import SectorNewsExtractor, sector_keywords
# from nlp.word_cloud import SectorWordCloud



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


def get_sector_dict():
    sectors = ['Materials', 'Industrials', 'Health Care', 'Real Estate', 'Consumer Discretionary', 'Financials', 
               'Utilities', 'Information Technology', 'Energy', 'Consumer Staples', 'Communication Services']
    
    path = "Anomalies Dataset"
    model_list = sorted([x.split('.')[0] for x in next(os.walk(path))[2]])
    sector_dict = {}

    for sector in sectors:
        tmp = []
        for model in model_list:
            model_data = pd.read_csv(f"{path}/{model}.csv", index_col='date')
            model_data.index = pd.to_datetime(model_data.index)

            sector_anomalies = (model_data.loc[:, sector]).astype(int).rename(model)
            tmp.append(sector_anomalies)

        data = pd.concat(tmp, axis=1)
        sector_dict[sector] = data

    return sector_dict

        





