import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HUGGED_API_KEY")

API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

headers = {"Authorization": f"Bearer {api_key}"}

def get_embedding(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    response_data = response.json()
    return response_data   # HF returns nested structure


sentences = [
    "The stock market rose today after positive earnings reports.",
    "A new smartphone was released with advanced camera features.",
    "Scientists discovered water on Mars.",
    "The local bakery introduced a new chocolate cake.",
    "Heavy rainfall caused flooding in several areas."
]

embeddings = []

for text in sentences:
    vector = get_embedding(text)
    embeddings.append(vector)

# Print results
for i, emb in enumerate(embeddings):
    print(f"\nSentence {i+1}: {sentences[i]}")
    print(f"→ Vector length: {len(emb)}")
    print(f"→ First 5 values: {emb}")

