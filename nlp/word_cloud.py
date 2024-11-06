import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from nltk.stem import PorterStemmer
import re

class SectorWordCloud:
    def __init__(self, articles, stop_word_path = "nlp/stopwords.txt"):
        self.articles = articles
        self.corpus = ""
        self.word_ls = []
        with open(stop_word_path, "r") as file:
            self.stopwords = [line.strip() for line in file]

    def generate_word_cloud(self, sector_name, instreamlit = False):
        self._combine_text()
        self._tokenize()
        self._visualize(sector_name, instreamlit)

    def _combine_text(self):
        for article in self.articles:
            self.corpus = self.corpus + " " +article.get("body", "").strip()
        self.corpus.strip()
        
    def _tokenize(self, use_stemming=True):
        stop_words = set(self.stopwords)
        stemmer = PorterStemmer() if use_stemming else None
        data = re.sub(r'[^\w\s]', '', self.corpus)
        data = word_tokenize(data)
        for word in data:
            if word.isalpha():
                processed_word = stemmer.stem(word.lower()) if stemmer else word.lower()
                if processed_word not in stop_words:
                    self.word_ls.append(processed_word)

    def _visualize(self, sector_name, instreamlit):
        all_text = " ".join(self.word_ls)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"Word Cloud for {sector_name} Sector")
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        
        if instreamlit:
            st.pyplot(fig)
        else:
            plt.show()



