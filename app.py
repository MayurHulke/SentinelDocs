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

# Set page configuration with wider layout
st.set_page_config(
    page_title="SentinelDocs", 
    page_icon="🙈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for modern UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #4F46E5;
        --primary-light: #818CF8;
        --secondary: #06B6D4;
        --text-dark: #1E293B;
        --text-light: #64748B;
        --bg-light: #F8FAFC;
        --bg-dark: #0F172A;
        --success: #10B981;
        --warning: #F59E0B;
        --error: #EF4444;
        --radius: 8px;
    }
    
    /* Base layout improvements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom card component */
    .card {
        border-radius: var(--radius);
        border: 1px solid #E2E8F0;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Headers styling */
    h1, h2, h3, h4 {
        color: var(--text-dark);
        font-weight: 600;
    }
    
    h1 {
        font-size: 2.25rem;
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: 1.75rem;
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    
    h3 {
        font-size: 1.25rem;
        margin-top: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--bg-light);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border-radius: var(--radius);
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-light);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border-radius: var(--radius);
    }
    
    /* Progress and spinners */
    .stProgress .st-bo {
        background-color: var(--primary);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: var(--radius);
    }
    
    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-primary {
        background-color: var(--primary);
        color: white;
    }
    
    .badge-secondary {
        background-color: var(--secondary);
        color: white;
    }
    
    .badge-success {
        background-color: var(--success);
        color: white;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border-color: #E2E8F0;
    }
    
    /* Logo and app header */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        margin-bottom: 2rem;
    }
    
    .logo-img {
        width: 120px;
        height: 120px;
        margin-bottom: 1rem;
    }
    
    /* Two-column layout */
    .two-column {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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

# Check Ollama availability
ollama_available = check_ollama_availability()

# Get available models
available_models = get_available_models()

# Render custom header with logo and title
st.markdown("""
<div class="app-header">
    <div class="logo-img" style="font-size: 80px; display: flex; justify-content: center; align-items: center;">
        📄
    </div>
    <h1>SentinelDocs</h1>
    <p style="color: var(--text-light); font-size: 1.2rem; margin-top: -0.5rem;">Your Private AI-Powered Document Analyst</p>
</div>
""", unsafe_allow_html=True)

# Show Ollama warning if not available
if not ollama_available:
    st.error("⚠️ Ollama service is not available. Please make sure the Ollama server is running locally.")

# Setup sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Model selection with icons and descriptions
    st.markdown("#### Select AI Model")
    model_descriptions = {
        "deepseek-r1:8b": "Best for detailed analysis",
        "mistral": "Good balance of speed and accuracy",
        "llama3": "Best for creative responses",
        "phi3:mini": "Fast and efficient"
    }
    
    # Check which models are available and add descriptions
    model_options = []
    for model in available_models:
        desc = model_descriptions.get(model, "")
        if desc:
            model_options.append(f"{model} - {desc}")
        else:
            model_options.append(model)
    
    selected_model_option = st.selectbox(
        "Choose a model:",
        options=model_options,
        index=0
    )
    
    # Extract just the model name from selection
    selected_model = selected_model_option.split(" - ")[0] if " - " in selected_model_option else selected_model_option
    
    st.markdown("---")
    
    # Question history with better styling
    st.markdown("### 💬 Question History")
    
    if st.session_state.user_questions:
        for i, q in enumerate(st.session_state.user_questions, 1):
            st.markdown(f"""<div style="margin-bottom: 8px; padding: 8px; background-color: #f1f5f9; border-radius: 4px;">
                <span style="font-size: 0.8rem; color: #64748b;">Q{i}:</span> {q}
            </div>""", unsafe_allow_html=True)
            
        # Clear history button
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state.user_questions = []
            st.experimental_rerun()
    else:
        st.write("No previous questions yet.")
    
    # Add version info at bottom
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>SentinelDocs v1.0</div>", unsafe_allow_html=True)

# Main content area with tabs
main_tabs = st.tabs(["📂 Documents", "❓ Ask Questions", "📊 Insights"])

with main_tabs[0]:  # Documents Tab
    st.markdown("### Upload Your Documents")
    
    # File upload with guidelines
    st.markdown("""
    <div class="card">
        <p>Upload PDF, DOCX, or TXT files for analysis. Your documents stay private and are processed locally.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )
    
    if uploaded_files:
        try:
            document_texts = extract_text_from_files(uploaded_files)
            
            if not document_texts:
                st.error("No text could be extracted from the uploaded files.")
            else:
                # Generate document statistics
                st.session_state.document_stats = analyze_document_stats(document_texts)
                
                st.success(f"✅ Successfully processed {len(document_texts)} document(s)")
                
                # Document stats cards in two columns
                st.markdown("### 📊 Document Statistics")
                
                # Create two columns
                cols = st.columns(len(document_texts) if len(document_texts) <= 3 else 3)
                
                for idx, (doc_name, stats) in enumerate(st.session_state.document_stats.items()):
                    col_idx = idx % len(cols)
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div class="card">
                            <h3>{doc_name}</h3>
                            <p><b>Words:</b> {stats['word_count']}</p>
                            <p><b>Characters:</b> {stats['char_count']}</p>
                            <p><b>Sentences:</b> {stats.get('sentences', 'N/A')}</p>
                            
                            <div style="margin-top: 10px;">
                                <p><b>Key Entities:</b></p>
                                <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                        """, unsafe_allow_html=True)
                        
                        # Add entity badges with different colors based on type
                        entity_colors = {
                            "PERSON": "#4F46E5",
                            "ORG": "#06B6D4", 
                            "GPE": "#10B981",
                            "DATE": "#F59E0B",
                            "MONEY": "#7C3AED",
                            "TIME": "#EC4899"
                        }
                        
                        if stats.get('entities'):
                            for entity_type, examples in stats['entities'].items():
                                if examples:
                                    color = entity_colors.get(entity_type, "#64748B")
                                    for example in examples[:3]:  # Limit to 3 examples per type
                                        st.markdown(f"""
                                        <span class="badge" style="background-color: {color};">
                                            {example}
                                        </span>
                                        """, unsafe_allow_html=True)
                        
                        st.markdown("</div></div></div>", unsafe_allow_html=True)
                
                # Document preview tab
                st.markdown("### 📄 Document Preview")
                with st.expander("Click to View Document Content"):
                    for doc_name, doc_text in document_texts.items():
                        st.markdown(f"**{doc_name}**")
                        st.text_area(
                            label=f"Content of {doc_name}", 
                            value=doc_text[:1000] + ("..." if len(doc_text) > 1000 else ""), 
                            height=150,
                            label_visibility="collapsed"
                        )
                
                # Add document comparison section if multiple docs
                if len(document_texts) >= 2:
                    st.markdown("### 🔄 Document Comparison")
                    st.markdown("""
                    <div class="card">
                        <p>Compare documents to find similarities and differences between their content.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("Compare Documents", key="compare_docs"):
                        with st.spinner("Analyzing document similarities and differences..."):
                            comparisons = compare_documents(document_texts, selected_model)
                            
                            if isinstance(comparisons, str):
                                st.error(comparisons)
                            else:
                                for pair, result in comparisons.items():
                                    with st.expander(f"📊 {pair}"):
                                        st.markdown(result)
        
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

with main_tabs[1]:  # Questions Tab
    if not uploaded_files:
        st.info("Please upload documents in the Documents tab first.")
    else:
        st.markdown("### 💡 Ask Questions About Your Documents")
        
        # UI for search settings
        st.markdown("""
        <div class="card">
            <p>Ask any question about your documents. SentinelDocs will find the most relevant information.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Document selection option
        search_option = st.radio(
            "Search method:",
            ["Search across all documents", "Search specific document"],
            horizontal=True
        )
        
        specific_doc = None
        if search_option == "Search specific document":
            specific_doc = st.selectbox(
                "Select document:",
                list(document_texts.keys())
            )
        
        # Question input with example questions
        st.markdown("#### Your Question")
        
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
        
        # Create clickable suggestion buttons using Streamlit
        if "selected_question" not in st.session_state:
            st.session_state.selected_question = ""
            
        # Handle suggestions with session state
        cols = st.columns(4)
        for i, question in enumerate(common_questions):
            col_idx = i % 4
            with cols[col_idx]:
                if st.button(question, key=f"q_{i}"):
                    st.session_state.selected_question = question
        
        # Main question input, pre-filled if a suggestion was clicked
        user_question = st.text_input("Type your question:", value=st.session_state.selected_question, key="question-input")
        
        # Clear the selected question once it's been used
        if user_question != st.session_state.selected_question and st.session_state.selected_question != "":
            st.session_state.selected_question = ""
        
        # Generate response
        analyze_button = st.button("🔍 Analyze Documents", use_container_width=True)
        
        # Auto-analyze when a suggestion is clicked (if user hasn't changed the question)
        auto_analyze = False
        if st.session_state.selected_question != "" and user_question == st.session_state.selected_question:
            auto_analyze = True
            
        if analyze_button or auto_analyze:
            if not user_question:
                st.error("Please enter a question.")
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
                
                # Clear selected question after use
                if st.session_state.selected_question != "":
                    st.session_state.selected_question = ""

                # Display response in a card
                st.markdown("### 💡 Answer")
                
                # Show sources and model
                st.markdown("""
                <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;">
                    <span class="badge badge-primary">Model: {}</span>
                """.format(selected_model), unsafe_allow_html=True)
                
                if sources:
                    for source in sources:
                        st.markdown(f"""<span class="badge badge-secondary">Source: {source}</span>""", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display the actual response in a card
                st.markdown(f"""
                <div class="card" style="background-color: #f8fafc;">
                    {response.replace('\n', '<br>')}
                </div>
                """, unsafe_allow_html=True)

with main_tabs[2]:  # Insights Tab
    if not uploaded_files:
        st.info("Please upload documents in the Documents tab first.")
    else:
        st.markdown("### 📊 Document Insights & Reports")
        
        st.markdown("""
        <div class="card">
            <p>Generate comprehensive reports and extract key insights from your documents.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Report generation options
        report_type = st.radio(
            "Report type:",
            ["Executive Summary", "Key Insights", "Full Analysis"],
            horizontal=True
        )
        
        report_prompts = {
            "Executive Summary": "Create a concise executive summary of the document in 3-5 bullet points",
            "Key Insights": "Summarize the key insights and takeaways from this document",
            "Full Analysis": "Perform a detailed analysis of the document including key points, entities, recommendations, and potential issues"
        }
        
        if st.button("📄 Generate Report", use_container_width=True):
            with st.spinner(f"Generating {report_type.lower()}... This may take a minute."):
                insights = {doc: generate_response(report_prompts[report_type], document_texts, doc, selected_model) for doc in document_texts}
                pdf_file = generate_pdf_report(insights)
                
            # Success message with download button
            st.success(f"✅ {report_type} generated successfully!")
            
            with open(pdf_file, "rb") as file:
                st.download_button(
                    label="📥 Download Report",
                    data=file,
                    file_name=f"SentinelDocs_{report_type.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
            # Clean up temp file
            try:
                os.remove(pdf_file)
            except:
                pass

# Footer with additional information
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0;">
    <p style="color: #64748b; font-size: 0.9rem;">
        SentinelDocs - Your documents never leave your machine. All processing happens locally.
    </p>
</div>
""", unsafe_allow_html=True)