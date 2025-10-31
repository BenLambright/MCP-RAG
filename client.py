import requests

SERVER_URL = "http://localhost:8080/rag"  # or wherever were hosting

def ask_rag(question: str, top_k: int = 3):
    resp = requests.post(SERVER_URL, json={"question": question, "top_k": top_k})
    return resp.json()["retrieved_context"]

if __name__ == "__main__":
    query = "What are some recent highlights of Together AI?"
    print(ask_rag(query))