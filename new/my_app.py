import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, UploadFile, File, Query
import pickle
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

word2vec_model = None
fasttext_model = None
tfidf_vectorizer = None
document_vectors = None
df = pd.DataFrame()  


def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'[^\w\s\d]', '', text) 
    return text.split()  


def train_word2vec_model(text_data):
    processed_texts = [preprocess_text(text) for text in text_data]
    model = Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=1, workers=4)
    return model


def train_fasttext_model(text_data):
    processed_texts = [preprocess_text(text) for text in text_data]
    model = FastText(sentences=processed_texts, vector_size=100, window=5, min_count=1, workers=4)
    return model


def train_tfidf_model(text_data):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_data)
    return vectorizer, vectors.toarray()


def get_document_vectors(model, text_data):
    vectors = []
    for text in text_data:
        words = preprocess_text(text)
        word_vecs = [model.wv[word] for word in words if word in model.wv]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)


def ensure_vector_dimensions(vectors, target_dim):
    return np.array([np.pad(vec, (0, max(0, target_dim - len(vec))), 'constant') if len(vec) < target_dim else vec[:target_dim] for vec in vectors])


@app.post("/train")
async def train_model(file: UploadFile = File(...), model_type: str = Query("word2vec", enum=["word2vec", "fasttext", "tfidf"])):
    global word2vec_model, fasttext_model, tfidf_vectorizer, document_vectors, df
    
    df = pd.read_csv(file.file)
    
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    
    if model_type == "word2vec":
        word2vec_model = train_word2vec_model(df['clinic_notes'].tolist())
        document_vectors = get_document_vectors(word2vec_model, df['clinic_notes'])
        
        document_vectors = ensure_vector_dimensions(document_vectors, word2vec_model.vector_size)
        
        with open("word2vec_model.pkl", "wb") as f:
            pickle.dump(word2vec_model, f)
        
        with open("df_word2vec.pkl", "wb") as f:
            pickle.dump(df, f)
    elif model_type == "fasttext":
        fasttext_model = train_fasttext_model(df['clinic_notes'].tolist())
        document_vectors = get_document_vectors(fasttext_model, df['clinic_notes'])
        
        document_vectors = ensure_vector_dimensions(document_vectors, fasttext_model.vector_size)
        
        with open("fasttext_model.pkl", "wb") as f:
            pickle.dump(fasttext_model, f)
        
        with open("df_fasttext.pkl", "wb") as f:
            pickle.dump(df, f)
    elif model_type == "tfidf":
        tfidf_vectorizer, document_vectors = train_tfidf_model(df['clinic_notes'].tolist())
        
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(tfidf_vectorizer, f)
        
        with open("df_tfidf.pkl", "wb") as f:
            pickle.dump(df, f)
    
    return {"status": f"{model_type.capitalize()} model and data saved successfully"}


@app.post("/load-model")
async def load_model(model_type: str = Query("word2vec", enum=["word2vec", "fasttext", "tfidf"])):
    global word2vec_model, fasttext_model, tfidf_vectorizer, document_vectors, df
    
    
    if model_type == "word2vec":
        with open("word2vec_model.pkl", "rb") as f:
            word2vec_model = pickle.load(f)
        with open("df_word2vec.pkl", "rb") as f:
            df = pickle.load(f)
        document_vectors = get_document_vectors(word2vec_model, df['clinic_notes'])
        document_vectors = ensure_vector_dimensions(document_vectors, word2vec_model.vector_size)
    elif model_type == "fasttext":
        with open("fasttext_model.pkl", "rb") as f:
            fasttext_model = pickle.load(f)
        with open("df_fasttext.pkl", "rb") as f:
            df = pickle.load(f)
        document_vectors = get_document_vectors(fasttext_model, df['clinic_notes'])
        document_vectors = ensure_vector_dimensions(document_vectors, fasttext_model.vector_size)
    elif model_type == "tfidf":
        with open("tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        with open("df_tfidf.pkl", "rb") as f:
            df = pickle.load(f)
        document_vectors = tfidf_vectorizer.transform(df['clinic_notes'].tolist()).toarray()
    
    return {"status": f"{model_type.capitalize()} model and data loaded successfully"}

@app.get("/search")
async def search(query: str, model_type: str = Query("word2vec", enum=["word2vec", "fasttext", "tfidf"]), top_n: int = 5):
    global word2vec_model, fasttext_model, tfidf_vectorizer, document_vectors, df
    if model_type == "word2vec" and word2vec_model is None:
        return {"error": "Word2Vec model is not loaded or trained"}
    elif model_type == "fasttext" and fasttext_model is None:
        return {"error": "FastText model is not loaded or trained"}
    elif model_type == "tfidf" and tfidf_vectorizer is None:
        return {"error": "TF-IDF model is not loaded or trained"}
    
    # Preprocess query and convert to vector
    if model_type == "word2vec" or model_type == "fasttext":
        model = word2vec_model if model_type == "word2vec" else fasttext_model
        query_vector = np.mean([model.wv[word] for word in preprocess_text(query) if word in model.wv], axis=0)
        
        # Handle case where query_vector might be empty
        if query_vector.size == 0:
            return {"error": "Query did not contain any words in the model's vocabulary"}
        
        # Ensure query_vector has the correct dimensions
        query_vector = query_vector.reshape(1, -1)  # Reshape to 2D array
    elif model_type == "tfidf":
        query_vector = tfidf_vectorizer.transform([query]).toarray()
    
    # Ensure document_vectors is 2D
    if document_vectors.ndim != 2:
        return {"error": "Document vectors did not form a 2D array. Check the preprocessing step."}
    
    # Calculate similarity between query and document vectors
    similarities = cosine_similarity(query_vector, document_vectors).flatten()
    
    # Add similarity scores to DataFrame
    df['similarity'] = similarities
    
    # Sort by similarity and select top_n
    top_results = df.nlargest(top_n, 'similarity')
    
    # Collect results
    results = []
    for _, row in top_results.iterrows():
        results.append({
            "Clinic Notes": row['clinic_notes'],
            "Similarity Score": round(row["similarity"], 2),
        })
    
    return {"query": query, "results": results}
