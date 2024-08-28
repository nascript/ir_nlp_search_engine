import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Query, UploadFile, File
import pickle

# Initialize FastAPI
app = FastAPI()

# Global variables for the model and document vectors
word2vec_model = None
document_vectors = None
df = pd.DataFrame()  # Initialize an empty DataFrame

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'\d+', '', text.lower())  # Lowercase and remove digits
    text = re.sub(r'[^\w\s]', '', text)      # Remove punctuation
    return text.split()                      # Tokenize

# Train Word2Vec model
def train_word2vec_model(text_data):
    processed_texts = [preprocess_text(text) for text in text_data]
    model = Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Function to get document vectors
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

def calculate_correlation(doc, query):
    query_words = set(preprocess_text(query))
    doc_words = set(preprocess_text(doc))
    
    correlation_score = len(query_words.intersection(doc_words)) / len(query_words)
    
    return round(correlation_score, 2)

def add_additional_correlations(row, query):
    # Additional correlations for more fields
    correlations = {}
    
    # Example fields to add as potential correlations
    fields = [
        'Age', 'Gender', 'Ethnicity', 'ChronicConditions', 'Allergies', 
        'FamilyHistory', 'MedicationsHistory', 'SmokingStatus', 'AlcoholUse', 
        'BMI', 'Hospitalizations', 'Surgeries', 'Immunizations', 'LabResults', 'PrimaryDiagnosis'
    ]
    
    query_words = set(preprocess_text(query))
    
    for field in fields:
        field_words = set(preprocess_text(str(row[field])))
        correlations[field] = round(len(query_words.intersection(field_words)) / len(query_words), 2)
    
    # Filter correlations with a score > 0
    return {k: v for k, v in correlations.items() if v > 0}

# Endpoint to train the model with uploaded CSV and save it
@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    global word2vec_model, document_vectors, df
    # Load CSV data
    df = pd.read_csv(file.file)
    # Train the model
    word2vec_model = train_word2vec_model(df['Clinical Note'].tolist())
    # Calculate document vectors
    document_vectors = get_document_vectors(word2vec_model, df['Clinical Note'])
    # Save the trained model and df
    with open("word2vec_model.pkl", "wb") as f:
        pickle.dump(word2vec_model, f)
    with open("df.pkl", "wb") as f:
        pickle.dump(df, f)
    return {"status": "Model and data saved successfully"}

# Endpoint to load the saved model
@app.post("/load-model")
async def load_model():
    global word2vec_model, document_vectors, df
    
    # Load the trained model
    with open("word2vec_model.pkl", "rb") as f:
        word2vec_model = pickle.load(f)
    
    # Load the DataFrame df
    with open("df.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Recalculate document vectors
    document_vectors = get_document_vectors(word2vec_model, df['Clinical Note'])
    
    return {"status": "Model and data loaded successfully"}

# Endpoint to search documents
@app.get("/search")
async def search(query: str, top_n: int = 5):
    global word2vec_model, document_vectors, df
    if word2vec_model is None or document_vectors is None:
        return {"error": "Model is not loaded or trained"}
    
    results = []
    # Preprocess query and convert to vector
    query_vector = np.mean([word2vec_model.wv[word] for word in preprocess_text(query) if word in word2vec_model.wv], axis=0)
    
    # Calculate similarity between query and document vectors
    similarities = cosine_similarity([query_vector], document_vectors).flatten()
    
    # Add similarity scores to DataFrame
    df['Similarity'] = similarities
    
    # Sort by similarity and select top_n
    top_results = df.nlargest(top_n, 'Similarity')
    
    # For each top result, calculate correlation information
    for _, row in top_results.iterrows():
        correlation_score = calculate_correlation(row["Clinical Note"], query)
        
        correlated_info = {
            "Medication": row["Medication"],
            "DoctorSpecialty": row["DoctorSpecialty"],
            "PrimaryCareProvider": row["PrimaryCareProvider"],
            "Correlation Score": correlation_score
        }
        
        additional_correlations = add_additional_correlations(row, query)
        
        # Merge the additional correlations into correlated_info if they exist
        if additional_correlations:
            correlated_info.update(additional_correlations)
        
        results.append({
            "Clinical Note": row["Clinical Note"],
            "Similarity Score": round(row["Similarity"], 2),
            "Correlated Information": correlated_info
        })
    
    return {"query": query, "results": results}
