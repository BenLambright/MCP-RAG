import requests

SERVER_URL = "http://localhost:8080/rag"  # or wherever were hosting

def ask_rag(question: str):
    top_k = 3
    resp = requests.post(SERVER_URL, json={"question": question, "top_k": top_k})
    return resp.json()

if __name__ == "__main__":
    query = "Is the Dormouse Hatter nice to Alice?"
    print(ask_rag(query))