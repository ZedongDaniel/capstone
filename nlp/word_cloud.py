import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

class SectorWordCloud:
    def __init__(self, articles):
        self.articles = articles
        self.text_data = ""

    def generate_word_cloud(self, sector_name, instreamlit = False):
        self._combine_text()
        self._tokenize_and_clean()
        self._visualize(sector_name, instreamlit)

    def _combine_text(self):
        for article in self.articles:
            self.text_data += " " + article.get("body", "").strip()
        
    def _tokenize_and_clean(self):
        tokens = word_tokenize(self.text_data)
        stop_words = set(stopwords.words("english"))
        cleaned_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
        
        self.text_data = " ".join(cleaned_tokens)

    def _visualize(self, sector_name, instreamlit):
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(self.text_data)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"Word Cloud for {sector_name} Sector")
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        
        if instreamlit:
            st.pyplot(fig)
        else:
            plt.show() 



