import os
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

EMBED_MODEL = "gemini-embedding-001"

def embedded(text):
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )
    return np.array(response["embedding"])

sentences = [
    "pawan kalyan is a hero, his latest film is OG.",
    "cinos is a good cricketer, He even plays smash karts good.",
    "Todays news, A private bus from kaveri Travels met with an accident, 19 people were dead and 20 people were injured.",
    "the animal lying on the sofa is my dog.",
    "Bengaluru is a good location to enjoy, It is located in karnataka"
]

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sentence_vector = [embedded(s) for s in sentences]

query = input("Enter your query: ")
query_vec = embedded(query)

similarities = [cosine(query_vec, sv) for sv in sentence_vector]

best_index = np.argmax(similarities)

print("Most similar sentence: ")
print(sentences[best_index])
print("\nSimilarity scores:", similarities)