# server.py
# lanchain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# general imports
import sys

query = sys.argv[1]

# Load your FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)  # adjust path
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Query example
# query = "What are some recent highlights of Together AI?"
docs = retriever.invoke(query)  # returns the same list of documents

print("Retrieved Documents:")
for doc in docs:
    print(f"- {doc.page_content}")