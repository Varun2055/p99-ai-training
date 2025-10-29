import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import faiss

load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

EMBED_MODEL = "models/text-embedding-004"

def get_embedding(text):
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )
    return np.array(response["embedding"], dtype=np.float32)

documents = [
    "Pawan Kalyan's new movie OG is the most awaited movie.",
    "Virat Kohli is one of the best batsmen in the world.",
    "A dog is sleeping on the sofa peacefully.",
    "Heavy rainfall caused flooding in Hyderabad yesterday.",
    "The iPhone 16 launched with new AI features."
]


embeddings = np.array([get_embedding(doc) for doc in documents])

dimension = embeddings.shape[1]  # embedding dimension (768 for Gemini)
print("Embedding shape:", embeddings.shape)

index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
index.add(embeddings)

print("Vector store created with", index.ntotal, "documents.")



def search(query, k=2):
    query_vec = get_embedding(query).reshape(1, -1)

    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        results.append(documents[idx])

    return results


queries = [
    "Tell me about a new movie release.",
    "Who is the best cricket player?",
    "What happened due to rain?",
    "dog on the bed"
]

for q in queries:
    print("\nQuery:", q)
    results = search(q, k=2)
    for r in results:
        print(" →", r)
