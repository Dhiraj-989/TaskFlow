import streamlit as st
import os
import numpy as np
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
import google.generativeai as genai
import base64

load_dotenv()

# ---------------- CONFIG -------------------
st.set_page_config(page_title="TaskFlow RAG Agent", layout="wide")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY missing. Add it in your .env file.", icon="🚨")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

EMBED_MODEL = "models/text-embedding-004"
GEN_MODEL = "gemini-2.5-flash"

# ----------------- FUNCTIONS -------------------

def extract_text_from_pdf(pdf_file):
    temp_path = "uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.read())
    return extract_text(temp_path)

def chunk_text(text, chunk_size=1200):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    all_embeddings = []
    for chunk in chunks:
        resp = genai.embed_content(
            model=EMBED_MODEL, content=chunk, task_type="semantic_similarity"
        )
        emb = np.array(resp["embedding"])
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        all_embeddings.append(emb.tolist())
    return all_embeddings

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve_chunks(query, chunk_db, k):
    resp = genai.embed_content(
        model=EMBED_MODEL, content=query, task_type="semantic_similarity"
    )
    query_emb = np.array(resp["embedding"])

    scores = [
        (cosine_similarity(query_emb, np.array(item["embedding"])), item["text"])
        for item in chunk_db
    ]

    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:k]

# ------------------- UI ----------------------

st.title("📚 TaskFlow – PDF Question Answering Agent (RAG)")
st.write("Ask questions based on your uploaded PDF using **Gemini 2.5 Flash**.")

st.sidebar.header("📄 PDF Controls")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF file:", type=["pdf"])

chunk_size = st.sidebar.slider("Chunk Size (words)", 500, 2000, 1200)
top_k = st.sidebar.slider("Top-k Chunks", 1, 10, 5)

if uploaded_pdf:
    with st.spinner("Extracting and processing PDF..."):
        text = extract_text_from_pdf(uploaded_pdf)
        chunks = chunk_text(text, chunk_size)
        embeddings = embed_chunks(chunks)

        st.session_state["db"] = [
            {"text": chunks[i], "embedding": embeddings[i]}
            for i in range(len(chunks))
        ]

    st.success("PDF Processed Successfully! 🎉")

    st.subheader("📘 PDF Preview")
    

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_data = f.read()
    base64_pdf = base64.b64encode(pdf_data).decode("utf-8")
    
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" height="600" type="application/pdf"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


    st.write(f"**Total Chunks Created:** {len(chunks)}")

if "db" in st.session_state:

    st.subheader("💬 Ask a Question")
    user_query = st.text_input("Enter your question about the PDF:")

    if st.button("Get Answer"):
        with st.spinner("Thinking... 🤖"):
            top_chunks = retrieve_chunks(user_query, st.session_state["db"], top_k)

            context = "\n\n".join([c[1] for c in top_chunks])

            prompt = f"""
Use ONLY this PDF context to answer:

Context:
{context}

Question:
{user_query}

If not in PDF, say:
"I could not find this in the PDF."
"""

            model = genai.GenerativeModel(GEN_MODEL)
            response = model.generate_content(prompt)

        st.subheader("🧠 Answer")
        st.info(response.text)

        st.subheader("📌 Context Used")
        for i, (score, chunk) in enumerate(top_chunks, 1):
            with st.expander(f"Chunk {i} (score: {score:.4f})"):
                st.write(chunk)



