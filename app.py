import streamlit as st
import os
import json
import fitz  # PyMuPDF for PDFs
from docx import Document
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import spacy
from fpdf import FPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load NLP model for entity extraction
nlp = spacy.load("en_core_web_sm")

# Load sentence-transformers model for AI embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast, and effective model

# Extract text from multiple documents
def extract_text_from_files(uploaded_files):
    documents = {}
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split(".")[-1].lower()
        if file_type == "pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
        elif file_type == "docx":
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file_type == "txt":
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error(f"Unsupported file format: {uploaded_file.name}")
            continue
        documents[uploaded_file.name] = text
    return documents

# Entity extraction for keywords
def extract_entities(text):
    doc = nlp(text)
    return {ent.text: ent.label_ for ent in doc.ents}

# AI-based summarization
def generate_summary(document_text):
    llm = Ollama(model="mistral")
    prompt = f"Summarize the following document in bullet points:\n\n{document_text[:5000]}"
    return llm(prompt)

# AI-powered Q&A
def generate_response(question, document_text):
    llm = Ollama(model="deepseek-r1:8b")
    prompt = PromptTemplate.from_template(
        """
        You are an AI assistant analyzing confidential company documents. Given the extracted text:
        {document_text}
        Answer the following question concisely while preserving document confidentiality:
        {question}
        """
    )
    return llm(prompt.format(document_text=document_text[:5000], question=question))

# Semantic Search with FAISS and Real Embeddings
def build_faiss_index(documents):
    """Convert document text to embeddings and build a FAISS index."""
    texts = list(documents.values())
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance (euclidean)
    index.add(embeddings)
    
    return index, texts, list(documents.keys())

def semantic_search(query, index, texts, doc_names):
    """Search for the most relevant document based on semantic meaning."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k=1)  # Find the top match
    matched_doc = doc_names[indices[0][0]]
    return matched_doc, texts[indices[0][0]]

# Generate PDF report
def generate_pdf_report(insights):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "AI Document Insights Report", ln=True, align="C")
    pdf.ln(10)

    for key, value in insights.items():
        pdf.multi_cell(0, 10, f"{key}: {value}")

    pdf.output("AI_Document_Insights.pdf")
    return "AI_Document_Insights.pdf"

# Streamlit UI
st.title("📄 AI-Powered Confidential Document Q&A (Local)")

uploaded_files = st.file_uploader("Upload multiple documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    document_texts = extract_text_from_files(uploaded_files)

    if document_texts:
        st.write("### 🔍 Extracted Content Preview")
        for doc_name, doc_text in document_texts.items():
            st.subheader(doc_name)
            st.text_area("Extracted Text", doc_text[:1000], height=200)  # Show first 1000 chars
        
        # Smart Keyword Extraction
        st.write("### 🏷️ Extracted Keywords & Entities")
        for doc_name, doc_text in document_texts.items():
            st.subheader(doc_name)
            st.json(extract_entities(doc_text))

        # AI Summaries
        st.write("### ✍️ AI-Generated Summaries")
        for doc_name, doc_text in document_texts.items():
            st.subheader(doc_name)
            summary = generate_summary(doc_text)
            st.success(summary)

        # Build FAISS Index with Real AI Embeddings
        index, texts, doc_names = build_faiss_index(document_texts)

        # User Question for AI
        st.write("### 🤖 Ask AI a Question")
        user_question = st.text_input("Enter your question:")

        if st.button("Get AI Answer"):
            matched_doc, relevant_text = semantic_search(user_question, index, texts, doc_names)
            response = generate_response(user_question, relevant_text)
            st.write(f"**AI Answer from:** {matched_doc}")
            st.success(response)

        # Generate PDF Report
        if st.button("📄 Generate PDF Report"):
            insights = {doc: generate_summary(text) for doc, text in document_texts.items()}
            pdf_file = generate_pdf_report(insights)
            with open(pdf_file, "rb") as file:
                st.download_button("Download Report", file, file_name="AI_Report.pdf", mime="application/pdf")