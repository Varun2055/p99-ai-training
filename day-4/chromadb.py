import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

genai.configure(api_key=api_key)

embed_model = genai.GenerativeModel("text-embedding-004")

def get_embedding(text):
    result = genai.embed_content(model=embed_model, content=text)
    return result["embedding"]  


documents = [
    "India won the cricket match yesterday.",
    "A new movie was released last Friday.",
    "Heavy rain caused floods in several areas.",
    "A dog was sleeping on the bed.",
    "SpaceX launched a new rocket into orbit."
]

embeddings = [get_embedding(doc) for doc in documents]


client = chromadb.Client()  

collection = client.create_collection(
    name="my_vector_store",
    metadata={"hnsw:space": "cosine"} 
)

# Create IDs
ids = [str(i) for i in range(len(documents))]

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids
)

print("Vector store created with", collection.count(), "documents.")


def search(query, k=2):
    qvec = get_embedding(query)
    result = collection.query(
        query_embeddings=[qvec],
        n_results=k
    )
    return result["documents"][0]

queries = [
    "Tell me about a new movie release.",
    "Who is the best cricket player?",
    "What happened due to rain?",
    "dog on the bed"
]

for q in queries:
    print("\n Query:", q)
    results = search(q, 2)
    for r in results:
        print(" →", r)
