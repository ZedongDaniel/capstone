import requests
import pandas as pd

class StockData:
    """
    A class to fetch and process stock data from the EOD Historical Data API.
    """
    def __init__(self, csv_file_path: str, api_key: str) -> None:
        """
        Initialize the StockData object by loading ticker symbols and sectors from a CSV file.
        """
        self.ticker_names = pd.read_csv(csv_file_path)['Symbol'].tolist()
        self.ticker_sectors = pd.read_csv(csv_file_path)['GICS Sector'].tolist()
        self.ticker_info = dict(zip(self.ticker_names, self.ticker_sectors))
        self.api_key = api_key
        self.base_url = 'https://eodhd.com/api/eod'

    def _fetch_data(self, ticker: str, period: str, start: str, end: str):
        '''
        private method to fetch stock data for a single ticker symbol from the API.
        '''
        url = f"{self.base_url}/{ticker}.US?period={period}&from={start}&to={end}&api_token={self.api_key}&fmt=json"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()  
        else:
            print(f"Failed to fetch data for {ticker}: {response.status_code}")
            return None
        

    def fetch_stock(self, ticker: str, period: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetches stock data for a single ticker and returns it as a pandas DataFrame.
        """
        print(f"fetch data for {ticker}...")
        result = self._fetch_data(ticker, period, start, end)
        df = pd.DataFrame(result)
        return df
    
    def fetch_stocks_by_sectors(self, sector: str, period: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetches stock data for all tickers in a specific GICS sector and returns a merged DataFrame.
        """
        results = {}
        for ticker, s in self.ticker_info.items():
            if s == sector:
                results[ticker] = pd.DataFrame(self._fetch_data(ticker, period, start, end))

        all_data = self._process_merge_all_data(results)
        return all_data
    
    def fetch_all_stocks(self, period: str, start: str, end: str):
        '''
        Fetches stock data for all tickers and returns a merged DataFrame.
        '''
        results = {}
        for ticker in self.ticker_info.keys():
            results[ticker] = pd.DataFrame(self._fetch_data(ticker, period, start, end))

        all_data = self._process_merge_all_data(results)
        return all_data

    def _process_merge_all_data(self, dic:dict) -> pd.DataFrame:
        '''
        Private method to process and merge stock data from multiple tickers into a single DataFrame.
        '''
        all_data = []

        for name, df in dic.items():
            if df.empty:
                print(f"no data for {name}. skipping...")
                continue

            df['date'] = pd.to_datetime(df['date'])
            df['ticker'] = name
            all_data.append(df)

        full_data = pd.concat(all_data)

        full_data.set_index(['date', 'ticker'], inplace=True)

        return full_data


        
        
