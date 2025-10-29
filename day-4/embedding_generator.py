import os
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    "Banglore is a good location to enjoy, It is located in karnataka"
]

embeddings = [embedded(text) for text in sentences]
embeddings = np.array(embeddings)

print("Embedding_shape:", embeddings.shape)

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(7, 7))
plt.scatter(reduced[:, 0], reduced[:, 1])

for i, text in enumerate(sentences):
    plt.annotate(f"S{i+1}", (reduced[i, 0], reduced[i, 1]))

plt.title("2D Visualization of Gemini Embeddings (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
