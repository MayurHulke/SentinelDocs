import streamlit as st
import os
import json
import fitz  # PyMuPDF for PDFs
from docx import Document
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import spacy
from fpdf import FPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load NLP model for NER
try:
    nlp = spacy.load("en_core_web_sm")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error loading NLP models: {str(e)}. Some features may be limited.")
    nlp = None
    embedding_model = None

# Initialize session state for query history
if "user_questions" not in st.session_state:
    st.session_state.user_questions = []
if "referenced_docs" not in st.session_state:
    st.session_state.referenced_docs = []
if "document_stats" not in st.session_state:
    st.session_state.document_stats = {}

# Check if Ollama server is available
def check_ollama_availability():
    try:
        llm = Ollama(model="deepseek-r1:8b")
        llm("Hi")
        return True
    except Exception:
        return False

# Get available Ollama models
def get_available_models():
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        output = result.stdout
        
        models = []
        # Parse the output to extract model names
        for line in output.split('\n')[1:]:  # Skip header line
            if line.strip():
                model_name = line.split()[0]
                if ":" in model_name:  # Only include models with tags
                    models.append(model_name)
        
        return models if models else ["deepseek-r1:8b"]  # Default if no models found
    except Exception:
        return ["deepseek-r1:8b"]  # Default fallback

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

# Create document index for semantic search
def create_document_index(documents):
    chunks = []
    chunk_sources = []
    chunk_size = 1000
    chunk_overlap = 200
    
    for doc_name, text in documents.items():
        # Simple text chunking - could be improved with more sophisticated methods
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 100:  # Ignore very small chunks
                chunks.append(chunk)
                chunk_sources.append(doc_name)
    
    # Create embeddings
    embeddings = embedding_model.encode(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return index, chunks, chunk_sources

# Search for relevant document chunks
def search_documents(query, index, chunks, chunk_sources, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "text": chunks[idx],
                "source": chunk_sources[idx],
                "score": float(distances[0][i])
            })
    
    return results

# Generate response using LLM with relevant contexts
def generate_response(question, documents, doc_name=None, model_name="deepseek-r1:8b"):
    try:
        # If a specific document is provided, use it directly
        if doc_name and doc_name in documents:
            document_text = documents[doc_name]
            context = document_text[:5000]  # Use first 5000 chars
        # Otherwise use semantic search to find relevant parts
        else:
            # Create or get index from session state
            if "document_index" not in st.session_state:
                index, chunks, chunk_sources = create_document_index(documents)
                st.session_state.document_index = index
                st.session_state.document_chunks = chunks
                st.session_state.chunk_sources = chunk_sources
            else:
                index = st.session_state.document_index
                chunks = st.session_state.document_chunks
                chunk_sources = st.session_state.chunk_sources
            
            # Search for relevant chunks
            results = search_documents(question, index, chunks, chunk_sources)
            context = "\n\n".join([r["text"] for r in results])
            
            # For traceability, store which documents were referenced
            referenced_docs = list(set([r["source"] for r in results]))
            st.session_state.referenced_docs = referenced_docs
        
        # Generate response
        llm = Ollama(model=model_name)
        prompt = PromptTemplate.from_template(
            """
            You are an AI assistant analyzing documents. Given the extracted text:
            {document_text}
            
            Answer concisely and accurately:
            {question}
            """
        )
        return llm(prompt.format(document_text=context, question=question))
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Generate PDF report
def generate_pdf_report(insights):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "SentinelDocs - Document Insights Report", ln=True, align="C")
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    for doc_name, insight in insights.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Document: {doc_name}", ln=True)
        pdf.set_font("Arial", "", 11)
        
        # Split long text into multiple lines
        pdf.multi_cell(0, 10, insight)
        pdf.ln(5)
        
    temp_file = "temp_report.pdf"
    pdf.output(temp_file)
    return temp_file

# Generate document statistics
def analyze_document_stats(documents):
    stats = {}
    
    for doc_name, text in documents.items():
        doc_stats = {
            "word_count": len(text.split()),
            "char_count": len(text),
            "entities": {},
            "sentences": 0
        }
        
        # Extract entities if NLP is available
        if nlp:
            try:
                doc = nlp(text[:100000])  # Limit to first 100k chars for performance
                doc_stats["sentences"] = len(list(doc.sents))
                
                # Count entity types
                entities = {}
                for ent in doc.ents:
                    if ent.label_ not in entities:
                        entities[ent.label_] = []
                    if len(entities[ent.label_]) < 5:  # Limit to 5 examples per type
                        entities[ent.label_].append(ent.text)
                        
                doc_stats["entities"] = entities
            except Exception as e:
                st.warning(f"Error analyzing entities in {doc_name}: {str(e)}")
        
        stats[doc_name] = doc_stats
    
    return stats

# Compare documents
def compare_documents(documents, model_name="deepseek-r1:8b"):
    if len(documents) < 2:
        return "Need at least two documents to compare."
    
    try:
        llm = Ollama(model=model_name)
        
        # Get pairs of documents
        doc_names = list(documents.keys())
        results = {}
        
        for i in range(len(doc_names)):
            for j in range(i+1, len(doc_names)):
                doc1 = doc_names[i]
                doc2 = doc_names[j]
                
                # Get summaries of each document
                summary1 = documents[doc1][:2000]  # Use first 2000 chars for comparison
                summary2 = documents[doc2][:2000]
                
                prompt = PromptTemplate.from_template(
                    """
                    Compare and contrast these two document excerpts:
                    
                    DOCUMENT 1: {doc1_name}
                    {doc1_text}
                    
                    DOCUMENT 2: {doc2_name}
                    {doc2_text}
                    
                    Provide a concise analysis of:
                    1. Key similarities
                    2. Notable differences
                    3. Any conflicting information
                    """
                )
                
                comparison = llm(prompt.format(
                    doc1_name=doc1,
                    doc1_text=summary1,
                    doc2_name=doc2,
                    doc2_text=summary2
                ))
                
                key = f"{doc1} vs {doc2}"
                results[key] = comparison
        
        return results
    except Exception as e:
        return f"Error comparing documents: {str(e)}"

# Configure Streamlit page layout
st.set_page_config(page_title="SentinelDocs", page_icon="🙈", layout="centered")

# Check Ollama availability
ollama_available = check_ollama_availability()
if not ollama_available:
    st.error("⚠️ Ollama service is not available. Please make sure the Ollama server is running locally.")

# Get available models
available_models = get_available_models()

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

# Setup sidebar
with st.sidebar:
    st.subheader("⚙️ Settings")
    
    # Model selection
    selected_model = st.selectbox(
        "Select AI Model:",
        available_models,
        index=0 if "deepseek-r1:8b" not in available_models else available_models.index("deepseek-r1:8b")
    )
    
    st.markdown("---")
    
    # Question history section will follow

# File upload interface
st.subheader("📂 Upload Your Documents")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

if uploaded_files:
    try:
        document_texts = extract_text_from_files(uploaded_files)
        
        if not document_texts:
            st.error("No text could be extracted from the uploaded files.")
        else:
            # Generate document statistics
            st.session_state.document_stats = analyze_document_stats(document_texts)
            
            st.markdown("---")
            
            # Display document stats
            st.subheader("📊 Document Statistics")
            for doc_name, stats in st.session_state.document_stats.items():
                with st.expander(f"📄 {doc_name} Stats"):
                    st.write(f"**Words:** {stats['word_count']}")
                    st.write(f"**Characters:** {stats['char_count']}")
                    if 'sentences' in stats:
                        st.write(f"**Sentences:** {stats['sentences']}")
                    
                    # Display entity information if available
                    if stats.get('entities'):
                        st.write("**Key Entities:**")
                        for entity_type, examples in stats['entities'].items():
                            if examples:
                                st.write(f"- {entity_type}: {', '.join(examples)}")
            
            # Display extracted text
            st.subheader("📖 Extracted Content")
            with st.expander("📄 Click to View Extracted Text"):
                for doc_name, doc_text in document_texts.items():
                    st.markdown(f"**{doc_name}**")
                    st.text_area(
                        label=f"Content of {doc_name}", 
                        value=doc_text[:1000], 
                        height=150,
                        label_visibility="collapsed"
                    )

            # Add document comparison section if multiple docs
            if len(document_texts) >= 2:
                st.markdown("---")
                st.subheader("🔄 Document Comparison")
                if st.button("Compare Documents", key="compare_docs"):
                    with st.spinner("Analyzing document similarities and differences..."):
                        comparisons = compare_documents(document_texts, selected_model)
                        
                        if isinstance(comparisons, str):
                            st.error(comparisons)
                        else:
                            for pair, result in comparisons.items():
                                with st.expander(f"📊 {pair}"):
                                    st.markdown(result)

            # Add document selection option
            st.markdown("---")
            st.subheader("🔍 Document Search Settings")
            search_option = st.radio(
                "How would you like to search?",
                ["Search across all documents", "Search specific document"]
            )
            
            specific_doc = None
            if search_option == "Search specific document":
                specific_doc = st.selectbox(
                    "Select document to query:",
                    list(document_texts.keys())
                )

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
                if not user_question:
                    st.error("Please enter a question or select a suggested one.")
                else:
                    # Display loading indicator
                    with st.spinner("Analyzing documents... This may take a moment."):
                        # Generate response 
                        if search_option == "Search specific document" and specific_doc:
                            response = generate_response(user_question, document_texts, specific_doc, selected_model)
                            sources = [specific_doc]
                        else:
                            # Use semantic search across all documents
                            response = generate_response(user_question, document_texts, None, selected_model)
                            sources = st.session_state.referenced_docs if "referenced_docs" in st.session_state else []

                # Append to query history
                st.session_state.user_questions.append(user_question)

                st.markdown("---")
                st.subheader("💡 Here's what we found")
                
                # Display sources and model used
                st.markdown(f"**Model Used:** {selected_model}")
                if sources:
                    st.markdown("**Sources:**")
                    for source in sources:
                        st.markdown(f"- {source}")
                    
                st.info(response)  # Display response

            # Display query history in sidebar
            with st.sidebar:
                st.subheader("💬 Question History")
                if st.session_state.user_questions:
                    for i, q in enumerate(st.session_state.user_questions, 1):
                        st.write(f"{i}. {q}")
                else:
                    st.write("No previous questions yet.")
                    
                # Add clear history button
                if st.session_state.user_questions and st.button("Clear History"):
                    st.session_state.user_questions = []
                    st.experimental_rerun()

            # Generate and download PDF report
            st.markdown("---")
            if st.button("📄 Download Insights Report", use_container_width=True):
                with st.spinner("Generating comprehensive report..."):
                    insights = {doc: generate_response("Summarize the key insights", document_texts, doc, selected_model) for doc in document_texts}
                    pdf_file = generate_pdf_report(insights)
                    
                with open(pdf_file, "rb") as file:
                    st.download_button("Download Report", file, file_name="AI_Report.pdf", mime="application/pdf")
                    
                # Clean up temp file
                try:
                    os.remove(pdf_file)
                except:
                    pass
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")