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


