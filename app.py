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

# Load NLP model for NER
nlp = spacy.load("en_core_web_sm")

# Load transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize session state for query history
if "user_questions" not in st.session_state:
    st.session_state.user_questions = []

# Extract text from uploaded files
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

# Generate response using LLM
def generate_response(question, document_text):
    llm = Ollama(model="deepseek-r1:8b")
    prompt = PromptTemplate.from_template(
        """
        You are an AI assistant analyzing documents. Given the extracted text:
        {document_text}
        Answer concisely and accurately:
        {question}
        """
    )
    return llm(prompt.format(document_text=document_text[:5000], question=question))

# Configure Streamlit page layout
st.set_page_config(page_title="SentinelDocs", page_icon="🙈", layout="centered")

# Render title and description
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>🙈 SentinelDocs</h1>
        <h4 style='color: gray;'>Your Private AI-Powered Document Analyst</h4>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# File upload interface
st.subheader("📂 Upload Your Documents")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

if uploaded_files:
    document_texts = extract_text_from_files(uploaded_files)

    if document_texts:
        st.markdown("---")
        
        # Display extracted text
        st.subheader("📖 Extracted Content")
        with st.expander("📄 Click to View Extracted Text"):
            for doc_name, doc_text in document_texts.items():
                st.markdown(f"**{doc_name}**")
                st.text_area("", doc_text[:1000], height=150)

        # Suggested questions for user
        st.markdown("---")
        st.subheader("💡 Ask Me Anything About Your Documents")

        common_questions = [
            "What are the key findings?",
            "Can you summarize the main points?",
            "Are there any important deadlines?",
            "What action items are recommended?",
            "Who are the key people mentioned?",
            "What financial or legal details are covered?",
            "Are there any risks or concerns?",
            "Does this document contain confidential data?"
        ]

        selected_question = st.selectbox(
            "Pick a suggested question or type your own:", ["Choose a question"] + common_questions
        )

        # User input for custom questions
        user_question = st.text_input(
            "Or enter your own question:", value=selected_question if selected_question != "Choose a question" else ""
        )

        # Generate and display AI response
        if st.button("🔍 Analyse", use_container_width=True):
            doc_name = list(document_texts.keys())[0]  # Assume first document is most relevant
            response = generate_response(user_question, document_texts[doc_name])

            # Append to query history
            st.session_state.user_questions.append(user_question)

            st.markdown("---")
            st.subheader("💡 Here's what we found")
            st.markdown(f"**From Document:** {doc_name}")
            st.info(response)  # Display response

        # Display query history in sidebar
        with st.sidebar:
            st.subheader("💬 Question History")
            if st.session_state.user_questions:
                for i, q in enumerate(st.session_state.user_questions, 1):
                    st.write(f"{i}. {q}")
            else:
                st.write("No previous questions yet.")

        # Generate and download PDF report
        st.markdown("---")
        if st.button("📄 Download Insights Report", use_container_width=True):
            insights = {doc: generate_response("Summarize the key insights", text) for doc, text in document_texts.items()}
            pdf_file = generate_pdf_report(insights)
            with open(pdf_file, "rb") as file:
                st.download_button("Download Report", file, file_name="AI_Report.pdf", mime="application/pdf")