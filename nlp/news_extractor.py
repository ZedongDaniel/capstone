from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from datetime import datetime
import pandas as pd

sector_keywords = {
    "Materials": ["mining", "chemical manufacturing", "raw materials"],
    "Industrials": ["manufacturing", "industrial equipment", "aerospace"],
    "Health Care": ["pharmaceuticals", "biotechnology", "health services"],
    "Real Estate": ["property development", "housing market", "commercial real estate"],
    "Consumer Discretionary": ["retail", "leisure products", "automobiles"],
    "Financials": ["banking", "financial services"],
    "Utilities": ["electricity", "natural gas", "water services"],
    "Information Technology": ["software", "hardware", "tech services"],
    "Energy": ["oil", "renewable energy", "gas"],
    "Consumer Staples": ["food products", "household goods", "beverages"],
    "Communication Services": ["telecom", "media", "advertising"]
}

class SectorNewsExtractor:
    def __init__(self, api_key, sector_keywords = None, general_keywords=None):
        self.er = EventRegistry(apiKey=api_key)
        self.sector_keywords = sector_keywords if sector_keywords else {}
        self.general_keywords = general_keywords if general_keywords else []
        self.market_sources = QueryItems.OR([
                                        "bloomberg.com",
                                        "cnbc.com",
                                        "reuters.com",
                                        "marketwatch.com",
                                        "ft.com",
                                        "wsj.com",
                                        "forbes.com",
                                        "businessinsider.com",
                                        "finance.yahoo.com",
                                        "economist.com"
                                    ])
        self.articles = []

    def fetch_articles(self, sector_name, date_start=None, date_end=None, max_articles=1):
        if sector_name not in self.sector_keywords:
            raise ValueError(f"Sector '{sector_name}' not defined in sector keywords.")
        
        sector_specific_keywords = self.sector_keywords[sector_name] + self.general_keywords
        date_start = date_start or datetime.now().strftime("%Y-%m-%d")
        date_end = date_end or datetime.now().strftime("%Y-%m-%d")
        
        query = QueryArticlesIter(
            keywords=QueryItems.OR(sector_specific_keywords),
            dateStart=date_start,
            dateEnd=date_end,
            sourceUri=self.market_sources,
            lang="eng",
        )

        for article in query.execQuery(self.er, sortBy="rel", maxItems=max_articles):
            self.articles.append({
                "title": article.get("title"),
                "body": article.get("body"),
                "source": article.get("source", {}).get("uri"),
                "sentiment" : article.get("sentiment"),
                "date": article.get("date"),
            })

    
    def get_articles(self):
        return self.articles
    
    def set_general_keywords(self, keywords):
        self.general_keywords = keywords

    def set_sector_keywords(self, sector_name, keywords):
        self.sector_keywords[sector_name] = keywords

    def get_summary_table(self):
        date_ls = []
        source_ls = []
        sentiment_ls = []
        for article in self.articles:
            date_ls.append(article['date'])
            source_ls.append(article['source'])
            sentiment_ls.append(article['sentiment'])

        df = pd.DataFrame({
        'news date': date_ls,
        'news source': source_ls,
        'news sentiment': sentiment_ls
        })
        return df


