import pandas as pd
import numpy as np
from stock_data import StockData

def equal_index_construction(data: pd.DataFrame):
    def log_ret(s:pd.Series):
        return np.log(s / s.shift(1))
    
    data['log_ret'] = data.groupby('ticker', group_keys=False)['adjusted_close'].apply(log_ret)

    df = data.dropna().copy()
    df['w'] = df.groupby('date', group_keys=False)['adjusted_close'].transform(lambda x : 1 / len(x))

    index = pd.DataFrame()
    index['open'] = df.groupby('date').apply(lambda x: x['open']@x['w'])
    index['high'] = df.groupby('date').apply(lambda x: x['high']@x['w'])
    index['low'] = df.groupby('date').apply(lambda x: x['low']@x['w'])
    index['close'] = df.groupby('date').apply(lambda x: x['close']@x['w'])
    index['adjusted_close'] = df.groupby('date').apply(lambda x: x['adjusted_close']@x['w'])
    index['volume'] = df.groupby('date').apply(lambda x: x['volume']@x['w'])
    index['log_ret'] = df.groupby('date').apply(lambda x: x['log_ret']@x['w'])

    return index

stock = StockData('sp_400_midcap.csv', '662166cb8e3d13.57537943')
# df = stock.fetch_all_stocks(period = 'd', start = '2000-01-01', end = '2024-8-30')

# mid_cap_index = equal_index_construction(df)
# mid_cap_index.to_csv('index_data/mid_cap_index.csv')

# stock_info = pd.read_csv("sp_400_midcap.csv")
# sector_ls = stock_info["GICS Sector"].unique().tolist()
# sector_tmp = []
# for sector in sector_ls:
#     df = stock.fetch_stocks_by_sectors(sector=sector, period = 'd', start = '2000-01-01', end = '2024-8-30')
#     sector_idx = equal_index_construction(df)['log_ret'].rename(sector)
#     sector_tmp.append(sector_idx)

# data = pd.concat(sector_tmp, axis=1)
# data.to_csv('index_data/mid_cap_all_sectors_ret.csv')

stock_info = pd.read_csv("sp_400_midcap.csv")
sector_ls = stock_info["GICS Sector"].unique().tolist()
sector_tmp = []
for sector in sector_ls:
    df = stock.fetch_stocks_by_sectors(sector=sector, period = 'd', start = '2000-01-01', end = '2024-8-30')
    sector_idx = equal_index_construction(df)['volume'].rename(sector)
    sector_tmp.append(sector_idx)

data = pd.concat(sector_tmp, axis=1)
data.to_csv('index_data/mid_cap_all_sectors_volume.csv')


