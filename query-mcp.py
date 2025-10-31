# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# Pydantic model for incoming queries
class Query(BaseModel):
    question: str
    top_k: int = 3  # number of documents to retrieve

# Load your FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)  # adjust path
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Query example
# query = "What are some recent highlights of Together AI?"
# docs = retriever.invoke(query)  # returns the same list of documents

# Print retrieved documents (context)
@app.post("/rag")
def rag_endpoint(query: Query):
    # Retrieve documents
    docs = retriever.invoke(query.question)  # v0.2+ syntax
    # Combine retrieved text
    context = "\n---\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}
