import streamlit as st
import os
import fitz  # PyMuPDF
import pickle
import faiss
import numpy as np
import pandas as pd
import subprocess
from sentence_transformers import SentenceTransformer
from io import BytesIO


# Constants
INDEX_FILE = "rag.index"
CHUNKS_FILE = "chunks.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "tinyllama"

# Load embedding model once
model = SentenceTransformer(EMBEDDING_MODEL)

def load_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# --- Load Excel ---
def load_excel(uploaded_file):
    text = ""
    # Read the BytesIO file into ExcelFile correctly
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        text += f"--- Sheet: {sheet_name} ---\n"
        text += df.astype(str).to_string(index=False)
        text += "\n\n"
    return text

# --- Sentence-Based Chunking ---
def chunk_text(text, chunk_size=300):
    sentences = text.split('.')
    chunks, current = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 < chunk_size:  # +1 for the period
            current += sentence + "."
        else:
            chunks.append(current.strip())
            current = sentence + "."
    if current:
        chunks.append(current.strip())
    return chunks

def embed_chunks(chunks):
    return model.encode(chunks)

def save_index(embeddings, chunks):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_index_and_chunks():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def query_index(query, index, chunks, top_k=3):
    query_emb = model.encode([query])
    D, I = index.search(query_emb, top_k)
    return [chunks[i] for i in I[0]]

def call_ollama_llm(prompt):
    result = subprocess.run(
        ["ollama", "run", LLM_MODEL],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()

# --- Streamlit UI ---
st.title("🔍 RAG Prototype with FAISS + TinyLLaMA")

menu = st.sidebar.radio("Choose Action", ["1️⃣ Upload Document", "2️⃣ Ask a Question"])

if menu == "1️⃣ Upload Document":
    uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xlsx"])
    if uploaded_file:
        st.info("Extracting and indexing...")
        if uploaded_file.name.endswith(".pdf"):
            text = load_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            text = load_excel(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        save_index(embeddings, chunks)
        st.success(" Document embedded and saved to FAISS index.")

elif menu == "2️⃣ Ask a Question":
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        st.error("⚠️ No index found. Please upload a document first.")
    else:
        question = st.text_input("Ask a question:")
        if question:
            index, chunks = load_index_and_chunks()
            retrieved = query_index(question, index, chunks)
            context = "\n".join(retrieved)

            st.subheader("📚 Retrieved Chunks")
            for i, chunk in enumerate(retrieved, 1):
                st.code(f"Chunk {i}: {chunk.strip()}")

            prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {question}\nAnswer:"
            st.info("💬 Querying TinyLLaMA...")
            answer = call_ollama_llm(prompt)

            st.subheader("🧠 LLM Response")
            st.write(answer)
