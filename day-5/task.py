import os
import re
import faiss
import numpy as np
import google.generativeai as genai
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

class DocumentPipeline:

    def __init__(self):
        self.metadata_store = []
        self.index = None
        self.embeddings = None

    def load_pdf(self, file_path):
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    def load_txt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def load_file(self, file_path):
        if file_path.endswith(".pdf"):
            return self.load_pdf(file_path)
        elif file_path.endswith(".txt"):
            return self.load_txt(file_path)
        else:
            raise ValueError("File must be .pdf or .txt")

    def clean_text(self, text):
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"Page\s*\d+", "", text) 
        text = re.sub(r"\s+", " ", text) 
        return text.strip()


    def chunk_text(self, text, chunk_size=400, overlap=50, source="unknown", title="Document"):
        chunks = []
        start = 0
        chunk_id = 1

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": source,
                "title": title
            })

            chunk_id += 1
            start = end - overlap

        return chunks

    def embed(self, text):
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return np.array(response["embedding"], dtype="float32")

    def create_embeddings(self, chunks):
        embeddings = []

        for chunk in chunks:
            emb = self.embed(chunk["text"])
            embeddings.append(emb)
            self.metadata_store.append(chunk)

        self.embeddings = np.array(embeddings)
        return self.embeddings

    def build_faiss_index(self):
        if self.embeddings is None:
            raise ValueError("No embeddings found. Generate embeddings first.")

        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.embeddings)

        self.index = index
        return index

    def search(self, query, k=3):
        if self.index is None:
            raise ValueError("FAISS index not built yet!")

        query_emb = self.embed(query).reshape(1, -1)

        distances, indices = self.index.search(query_emb, k)

        results = []
        for idx in indices[0]:
            results.append(self.metadata_store[idx])

        return results


if __name__ == "__main__":
    pipeline = DocumentPipeline()

    raw_text = pipeline.load_file("demo.txt")

    cleaned_text = pipeline.clean_text(raw_text)

    chunks = pipeline.chunk_text(
        cleaned_text,
        chunk_size=400,
        overlap=50,
        source="demo.txt",
        title="My Demo Document"
    )

    pipeline.create_embeddings(chunks)

    pipeline.build_faiss_index()

    print("Pipeline Ready!")

    query = "What is the investigation about tell in 20 words?"
    results = pipeline.search(query)

    print("\n Search Results:")
    for r in results:
        print("-------------------------------------")
        print("Chunk ID:", r["chunk_id"])
        print("Text:", r["text"][:200], "...")
