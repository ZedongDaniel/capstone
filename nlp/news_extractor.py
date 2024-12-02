from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from datetime import datetime
import pandas as pd

class SectorNewsExtractor:
    def __init__(self, api_key):
        self.er = EventRegistry(apiKey=api_key)

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

    def fetch_articles(self, keyword, date_start=None, date_end=None, max_articles=1):
        date_start = date_start or datetime.now().strftime("%Y-%m-%d")
        date_end = date_end or datetime.now().strftime("%Y-%m-%d")
        
        query = QueryArticlesIter(
            keywords=keyword,
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


