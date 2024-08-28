import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import nltk
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize
from itertools import chain

# Download stopwords from nltk if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

class DocumentModel:
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.doc_vectors = None
        self.documents = []

        # Use stopwords from nltk
        self.stop_words = set(stopwords.words('english'))

    def simple_lemmatizer(self, word):
        if word.endswith('ing') or word.endswith('ed'):
            return word.rstrip('ing').rstrip('ed')
        return word

    def preprocess(self, doc):
        doc = re.sub(r'\d+', '', doc.lower())
        doc = re.sub(r'[^\w\s]', '', doc)
        tokens = doc.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.simple_lemmatizer(word) for word in tokens]
        return tokens

    def load_data_from_csv(self, csv_file, text_column):
        """Loads data from a CSV file and returns the documents as a list."""
        df = pd.read_csv(csv_file)
        self.documents = df[text_column].dropna().tolist()

    def train(self, documents=None):
        if documents:
            self.documents.extend(documents)
        preprocessed_docs = [self.preprocess(doc) for doc in self.documents]
        self.model = Word2Vec(sentences=preprocessed_docs, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=4)
        self.doc_vectors = np.array([self.document_vector(doc) for doc in self.documents])

    def document_vector(self, doc):
        words = self.preprocess(doc)
        word_vecs = [word for word in words if word in self.model.wv]
        if not word_vecs:
            return np.zeros(self.model.vector_size)
        return np.mean([self.model.wv[word] for word in word_vecs], axis=0)

    def find_similar_articles(self, query, top_n=3):
        query_vector = self.document_vector(query)
        cosine_similarities = cosine_similarity([query_vector], self.doc_vectors).flatten()
        similar_docs = cosine_similarities.argsort()[-top_n:][::-1]
        return [(self.documents[index], cosine_similarities[index]) for index in similar_docs]

    def eda(self):
        all_words = [word for doc in self.documents for word in self.preprocess(doc)]
        common_words = Counter(all_words).most_common(20)
        print("Most common words:", common_words)

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        plt.show()

        similarity_matrix = cosine_similarity(self.doc_vectors)
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='viridis')
        plt.title('Document Similarity Heatmap')
        plt.show()

    def update_model(self, new_documents):
        """Update the model with new documents."""
        self.documents.extend(new_documents)
        preprocessed_docs = [self.preprocess(doc) for doc in new_documents]
        self.model.build_vocab(preprocessed_docs, update=True)
        self.model.train(preprocessed_docs, total_examples=len(preprocessed_docs), epochs=self.model.epochs)
        new_doc_vectors = np.array([self.document_vector(doc) for doc in new_documents])
        self.doc_vectors = np.vstack((self.doc_vectors, new_doc_vectors))

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = Word2Vec.load(model_path)
        self.doc_vectors = np.array([self.document_vector(doc) for doc in self.documents])


