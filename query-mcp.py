# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

app = FastAPI()

# simple pydantic model to ensure the query passes, standard when dealing with APIs like this
class Query(BaseModel):
    question: str
    top_k: int = 3  # number of documents to retrieve

# booting up the corpus, but you can bring in a bigger document too
file_name = "Alices-Adventures-in-Wonderland-by-Lewis-Carroll.txt"
# Use os.path.join to construct the absolute path from the root of your app directory
file_path = os.path.join("/app", file_name) 

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# simple function to chunk the text so we have smaller pieces to embedd to capture information
def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Chunk the text and remove empty chunks
chunks_list = [chunk for chunk in split_text(text) if chunk.strip()]

# comput the TF-IDF embeddings but this time we don't have to do it from scratch
# I used sklearn's TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(chunks_list)
embeddings_list = [tfidf_matrix[i].toarray().flatten() for i in range(len(chunks_list))]

# rag function that can be called by the API
@app.post("/rag")
def rag_endpoint(query: Query):
    # Compute query embedding
    query_embedding = vectorizer.transform([query.question]).toarray().flatten()

    # Compute cosine similarities
    results = []
    for idx, chunk_emb in enumerate(embeddings_list):
        chunk_emb = np.array(chunk_emb)
        norm_chunk = np.linalg.norm(chunk_emb)
        norm_query = np.linalg.norm(query_embedding)
        sim = float(np.dot(chunk_emb, query_embedding) / (norm_chunk * norm_query)) if norm_chunk > 0 and norm_query > 0 else 0.0
        results.append({"doc_id": idx, "score": sim, "chunk_text": chunks_list[idx]})

    # Sort and return top_k results
    top_results = sorted(results, key=lambda x: x["score"], reverse=True)[:query.top_k]
    return top_results
