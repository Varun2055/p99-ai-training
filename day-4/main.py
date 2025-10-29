import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

genai.configure(api_key=api_key)

EMBED_MODEL = "models/text-embedding-004"

# --- function to get embedding ---
def embed(text):
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )
    return np.array(response["embedding"])

# --- cosine similarity ---
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- database sentences ---
sentences = [
    "A cat is sleeping on the sofa.",
    "The weather is sunny today.",
    "Dogs are loyal animals.",
    "I love eating fresh apples."
]

# --- embed all sentences ---
sentence_vectors = [embed(s) for s in sentences]

# --- user query ---
query = "how is the climate"
query_vec = embed(query)

# --- find most similar sentence ---
similarities = [cosine(query_vec, sv) for sv in sentence_vectors]

best_index = np.argmax(similarities)

print("Query:", query)
print("\nMost similar sentence:")
print(sentences[best_index])
print("\nSimilarity scores:", similarities)


# import os
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# api_key = os.getenv("API_KEY")
# if not api_key:
#     raise ValueError("API_KEY not found")

# genai.configure(api_key=api_key)

# EMBED_MODEL = "models/text-embedding-004"

# def get_embedding(text: str):
#     response = genai.embed_content(
#         model=EMBED_MODEL,
#         content=text
#     )
#     return response["embedding"] 

# text = "Hello, this is my first embedding!"
# vector = get_embedding(text)

# print("Embedding length:", len(vector))
# print(vector) 


# import os
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv("API_KEY")
# if not api_key:
#     raise ValueError("Key not found")

# genai.configure(api_key=api_key)

# model = genai.GenerativeModel("gemini-2.5-flash")

# EMBED_MODEL = "models/embedding-001"

# sentences = [
#     "The stock market rose today after positive earnings reports.",
#     "A new smartphone was released with advanced camera features.",
#     "Scientists discovered water on Mars.",
#     "The local bakery introduced a new chocolate cake.",
#     "Heavy rainfall caused flooding in several areas."
# ]

# embeddings = []

# def get_embedding(text):
#     response = genai.embed_text(
#         model=EMBED_MODEL,
#         text=text
#     )
#     return response['embedding']

# for text in sentences:
#     vector = get_embedding(text)
#     embeddings.append(vector)

# for i, emb in enumerate(embeddings):
#     print(f"\nSentence {i+1}: {sentences[i]}")
#     print(f"→ Vector length: {len(emb)}")
#     print(f"→ First 5 values: {emb[:5]}")