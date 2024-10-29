from news_extractor import SectorNewsExtractor
from word_cloud import SectorWordCloud

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

sector = 'Financials'
news_extractor = SectorNewsExtractor(api_key, sector_keywords, general_keywords=["SP400 Mid Cap", "mid-cap stocks"])
news_extractor.fetch_articles(sector_name=sector, date_start="2024-10-11", date_end="2024-10-11", max_articles=2)
articles = news_extractor.get_articles()

world_cloud = SectorWordCloud(articles)
world_cloud.generate_word_cloud(sector, instreamlit=False)