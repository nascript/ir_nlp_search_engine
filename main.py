# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from typing import List
from document_model import DocumentModel
import os

app = FastAPI()

# Initialize your document model
document_model = DocumentModel()

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # Load data using the correct column name
        document_model.load_data_from_csv(file_location, text_column='Clinical Note')
        document_model.train()
        document_model.save_model('word2vec.model')
        os.remove(file_location)
        return {"message": "Model trained and saved successfully"}
    else:
        return {"error": "Invalid file format. Please upload a CSV file."}

@app.post("/update")
async def update_model(docs: List[str]):
    if not document_model.model:
        document_model.load_model('word2vec.model')
    document_model.update_model(docs)
    document_model.save_model('word2vec.model')
    return {"message": "Model updated successfully"}

@app.post("/find_similar")
async def find_similar_articles(query: str, top_n: int = 3):
    if not document_model.model:
        document_model.load_model('word2vec.model')
    results = document_model.find_similar_articles(query, top_n)
    
    # Convert results to a format that can be returned by FastAPI
    formatted_results = [
        {"document": doc, "similarity": float(similarity)}
        for doc, similarity in results
    ]
    
    return {"results": formatted_results}


@app.get("/eda")
async def perform_eda():
    if not document_model.model:
        document_model.load_model('word2vec.model')
    document_model.eda()
    return {"message": "EDA performed and visualized"}
