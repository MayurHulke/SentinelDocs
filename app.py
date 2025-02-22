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

# **Initialize Session State for Question History**
if "user_questions" not in st.session_state:
    st.session_state.user_questions = []  # Store all past user questions

# **Extract Text from Multiple Documents**
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

# **AI-Powered Q&A**
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

# **Streamlit UI**
st.title("🙈 SentinelDocs")

uploaded_files = st.file_uploader("Upload multiple documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    document_texts = extract_text_from_files(uploaded_files)

    if document_texts:
        st.write("### 🔍 Extracted Content Preview")
        for doc_name, doc_text in document_texts.items():
            st.subheader(doc_name)
            st.text_area("Extracted Text", doc_text[:1000], height=200)  # Show first 1000 chars

        # **Suggested Questions**
        common_questions = [
            "What are the key findings of this document?",
            "Can you summarize the main points?",
            "Are there any important deadlines or dates mentioned?",
            "What action items or recommendations are in this document?",
            "Who are the key people or organizations referenced?",
            "What financial or legal details are covered?",
            "Are there any risks or concerns discussed?",
            "Does this document contain confidential or sensitive information?"
        ]

        st.write("### 💡 Suggested Questions")
        selected_question = st.selectbox("Select a suggested question or type your own:", ["Choose a question"] + common_questions)

        # **User Question Input**
        user_question = st.text_input("Or enter your own question:", value=selected_question if selected_question != "Choose a question" else "")

        if st.button("Get AI Answer"):
            doc_name = list(document_texts.keys())[0]  # Assume first document is most relevant
            response = generate_response(user_question, document_texts[doc_name])

            # **Save the user's question to history**
            st.session_state.user_questions.append(user_question)

            st.write(f"**AI Answer from:** {doc_name}")
            st.success(response)

        # **Show User's Question History (Now Stores All Questions)**
        with st.expander("📝 Previous Questions"):
            if st.session_state.user_questions:
                for i, q in enumerate(st.session_state.user_questions, 1):
                    st.write(f"{i}. {q}")
            else:
                st.write("No previous questions yet.")

        # **Generate PDF Report**
        if st.button("🖇️ Generate PDF Report"):
            insights = {doc: generate_response("Summarize the key insights", text) for doc, text in document_texts.items()}
            pdf_file = generate_pdf_report(insights)
            with open(pdf_file, "rb") as file:
                st.download_button("Download Report", file, file_name="AI_Report.pdf", mime="application/pdf")