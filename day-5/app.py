import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
import re
from pypdf import PdfReader
import nltk
nltk.download('punkt')

# ----------------------------------------------
# CONFIG
# ----------------------------------------------
genai.configure(api_key="YOUR_API_KEY")
EMBED_MODEL = "models/embedding-001"


# ----------------------------------------------
# CLEANING
# ----------------------------------------------
def clean_text(text: str):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ----------------------------------------------
# CHUNKING
# ----------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap  # move pointer with overlap

    return chunks


# ----------------------------------------------
# EMBEDDINGS
# ----------------------------------------------
def get_embeddings(chunks):
    embeddings = genai.embed_content(
        model=EMBED_MODEL,
        content=chunks,
        task_type="retrieval_document"
    )["embedding"]
    return np.array(embeddings).astype("float32")


# ----------------------------------------------
# BUILD FAISS
# ----------------------------------------------
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# ----------------------------------------------
# SEARCH
# ----------------------------------------------
def search(query, chunks, index, k=3):
    query_emb = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    query_emb = np.array(query_emb).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_emb, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "chunk": chunks[idx],
            "distance": float(score)
        })

    return results


# ----------------------------------------------
# STREAMLIT UI
# ----------------------------------------------
st.title("📚 AI Document Chunking + Vector Search (FAISS + Gemini)")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

# Sidebar settings
st.sidebar.title("⚙️ Settings")
chunk_size = st.sidebar.selectbox("Chunk Size", [300, 400, 500])
overlap = st.sidebar.slider("Overlap (words)", 0, chunk_size - 1, 50)

if uploaded_file:
    # Extract text
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    else:
        text = uploaded_file.read().decode("utf-8")

    st.subheader("✅ Raw Text")
    st.text_area("Raw", text, height=200)

    # Clean
    cleaned = clean_text(text)
    st.subheader("✅ Cleaned Text")
    st.text_area("Cleaned", cleaned, height=200)

    # Chunk the text
    chunks = chunk_text(cleaned, chunk_size, overlap)
    st.subheader(f"✅ Chunks Created — {len(chunks)} chunks")
    st.write(chunks[:3])  # preview

    if st.button("Generate Embeddings + Build Index"):
        with st.spinner("Generating embeddings using Gemini..."):
            embeddings = get_embeddings(chunks)

        st.success(f"Embeddings generated! Shape: {embeddings.shape}")

        index = build_faiss_index(embeddings)

        # Store
        st.session_state["index"] = index
        st.session_state["chunks"] = chunks

        st.success("FAISS index built successfully!")


# ----------------------------------------------
# SEARCH SECTION
# ----------------------------------------------
st.subheader("🔍 Semantic Search")
query = st.text_input("Enter your search query")

if query:
    if "index" not in st.session_state:
        st.error("Please generate embeddings first!")
    else:
        results = search(query, st.session_state["chunks"], st.session_state["index"])

        st.subheader("✅ Top Matches")
        for r in results:
            st.write(f"**Distance:** {r['distance']:.4f}")
            st.info(r["chunk"])
