from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from datetime import datetime

class SectorNewsExtractor:
    def __init__(self, api_key, sector_keywords = None, general_keywords=None):
        self.er = EventRegistry(apiKey=api_key)
        self.sector_keywords = sector_keywords if sector_keywords else {}
        self.general_keywords = general_keywords if general_keywords else []
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
            lang="eng"
        )

        for article in query.execQuery(self.er, sortBy="rel", maxItems=max_articles):
            self.articles.append({
                "title": article.get("title"),
                "body": article.get("body"),
                "source": article.get("source", {}).get("uri"),
                "sentiment" : article.get("sentiment"),
                "date": article.get("date"),
            })

        return None
    
    def get_articles(self):
        return self.articles
    
    def set_general_keywords(self, keywords):
        self.general_keywords = keywords

    def set_sector_keywords(self, sector_name, keywords):
        self.sector_keywords[sector_name] = keywords


if __name__ == "__main__":
    api_key = "cfd38812-1b95-4114-b0c0-39efacba95cf"
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

    news_extractor = SectorNewsExtractor(api_key, sector_keywords, general_keywords=["SP400 Mid Cap", "mid-cap stocks"])
    news_extractor.fetch_articles(sector_name="Financials", date_start="2024-10-11", date_end="2024-10-11", max_articles=2)
    articles = news_extractor.get_articles()
    for article in articles:
        print("title:", article["title"])
        print("body:", article["body"])
        print("source:", article["source"])
        print("sentiment:", article["sentiment"])
        print("date:", article["date"])
        print("-" * 80)
