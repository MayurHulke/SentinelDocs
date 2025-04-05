"""
Main application module for SentinelDocs.

This module provides the Streamlit web interface for the application.
"""

import streamlit as st
import os
from typing import Dict, List, Any, Optional, Tuple

# Import SentinelDocs modules
from sentineldocs.utils.config import get_config
from sentineldocs.utils.logging import setup_logging, get_logger
from sentineldocs.document import extract_text_from_files, analyze_document_stats, load_nlp_model
from sentineldocs.analyze import (
    generate_response, compare_documents, create_document_index, 
    search_documents, load_embedding_model
)
from sentineldocs.utils.pdf import generate_pdf_report
from sentineldocs.ui.styles import MAIN_CSS, HEADER_HTML, FOOTER_HTML
from sentineldocs.ui.components import (
    document_status_card, question_suggestions, source_badge, 
    model_badge, response_card, create_tab_content
)

# Load configuration
config = get_config()

# Set up logging
logger = setup_logging()
app_logger = get_logger("app")

def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "user_questions" not in st.session_state:
        st.session_state.user_questions = []
    if "referenced_docs" not in st.session_state:
        st.session_state.referenced_docs = []
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = {}
    if "document_index" not in st.session_state:
        st.session_state.document_index = None
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = []
    if "chunk_sources" not in st.session_state:
        st.session_state.chunk_sources = []
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = ""

def check_ollama_availability() -> bool:
    """
    Check if Ollama server is available.
    
    Returns:
        bool: True if Ollama is available, False otherwise
    """
    from langchain_ollama import OllamaLLM
    try:
        app_logger.info("Checking Ollama availability")
        default_model = config.get("ai.default_model", "deepseek-r1:8b")
        llm = OllamaLLM(model=default_model)
        llm.invoke("Hi")
        app_logger.info("Ollama is available")
        return True
    except Exception as e:
        app_logger.error(f"Ollama not available: {str(e)}")
        return False

def get_available_models() -> List[str]:
    """
    Get list of available Ollama models.
    
    Returns:
        List[str]: List of available model names
    """
    try:
        app_logger.info("Getting available Ollama models")
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
        
        default_model = config.get("ai.default_model", "deepseek-r1:8b")
        return models if models else [default_model]
    except Exception as e:
        app_logger.error(f"Error getting models: {str(e)}")
        return [config.get("ai.default_model", "deepseek-r1:8b")]

def setup_page() -> None:
    """Configure the Streamlit page settings."""
    app_logger.info("Setting up Streamlit page")
    st.set_page_config(
        page_title=config.get("app.title", "SentinelDocs"),
        page_icon=config.get("app.logo_emoji", "📄"),
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown(MAIN_CSS, unsafe_allow_html=True)
    
    # Render custom header with logo and title
    st.markdown(HEADER_HTML, unsafe_allow_html=True)

def setup_sidebar(available_models: List[str]) -> str:
    """
    Set up the sidebar with controls and display elements.
    
    Args:
        available_models: List of available Ollama models
        
    Returns:
        str: The selected model name
    """
    app_logger.info("Setting up sidebar")
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model selection with descriptions
        st.markdown("#### Select AI Model")
        model_descriptions = config.get("ai.model_descriptions", {})
        
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
        version = config.get("app.version", "1.0.0")
        st.markdown(f'<div style="text-align: center; color: #94a3b8; font-size: 0.8rem;">{config.get("app.title", "SentinelDocs")} v{version}</div>', unsafe_allow_html=True)
    
    return selected_model

def documents_tab(document_texts: Optional[Dict[str, str]], nlp) -> None:
    """
    Render the documents tab with file upload and document analysis.
    
    Args:
        document_texts: Dictionary of document texts (optional)
        nlp: spaCy language model for NLP tasks
    """
    app_logger.info("Rendering documents tab")
    st.markdown("### Upload Your Documents")
    
    # File upload with guidelines
    st.markdown("""
    <div class="card">
        <p>Upload PDF, DOCX, or TXT files for analysis. Your documents stay private and are processed locally.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=config.get("document.supported_formats", ["pdf", "docx", "txt"]), 
        accept_multiple_files=True
    )
    
    # Return if no files uploaded
    if not uploaded_files:
        return None
    
    # Process uploaded files
    try:
        app_logger.info(f"Processing {len(uploaded_files)} uploaded files")
        document_texts = extract_text_from_files(uploaded_files)
        
        if not document_texts:
            st.error("No text could be extracted from the uploaded files.")
            return None
            
        # Generate document statistics
        app_logger.info("Generating document statistics")
        st.session_state.document_stats = analyze_document_stats(document_texts, nlp)
        
        st.success(f"✅ Successfully processed {len(document_texts)} document(s)")
        
        # Document stats cards in columns
        st.markdown("### 📊 Document Statistics")
        
        # Create columns
        cols = st.columns(len(document_texts) if len(document_texts) <= 3 else 3)
        
        # Show document stats in cards
        for idx, (doc_name, stats) in enumerate(st.session_state.document_stats.items()):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                document_status_card(doc_name, stats)
        
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
                    app_logger.info("Comparing documents")
                    model = st.session_state.get("selected_model", config.get("ai.default_model", "deepseek-r1:8b"))
                    comparisons = compare_documents(document_texts, model)
                    
                    if isinstance(comparisons, dict) and comparisons.get("error"):
                        st.error(comparisons["error"])
                    else:
                        for pair, result in comparisons.items():
                            with st.expander(f"📊 {pair}"):
                                st.markdown(result)
    
    except Exception as e:
        app_logger.error(f"Error processing documents: {str(e)}")
        st.error(f"Error processing documents: {str(e)}")
        return None
        
    return document_texts

def questions_tab(document_texts: Optional[Dict[str, str]], selected_model: str) -> None:
    """
    Render the questions tab for document queries.
    
    Args:
        document_texts: Dictionary of document texts
        selected_model: Selected Ollama model name
    """
    app_logger.info("Rendering questions tab")
    
    if not document_texts:
        st.info("Please upload documents in the Documents tab first.")
        return
        
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
    
    # Get common questions from config
    common_questions = config.get("questions.common", [
        "What are the key findings?",
        "Can you summarize the main points?",
        "Are there any important deadlines?",
        "What action items are recommended?"
    ])
    
    # Function to set selected question
    def on_question_select(question: str) -> None:
        app_logger.info(f"Question selected: {question}")
        st.session_state.selected_question = question
    
    # Create clickable suggestion buttons
    question_suggestions(common_questions, on_question_select)
    
    # Main question input, pre-filled if a suggestion was clicked
    user_question = st.text_input(
        "Type your question:", 
        value=st.session_state.selected_question, 
        key="question-input"
    )
    
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
                app_logger.info(f"Generating response for question: {user_question}")
                
                # Get or create document index for semantic search
                embedding_model = load_embedding_model()
                
                if "document_index" not in st.session_state or st.session_state.document_index is None:
                    app_logger.info("Creating document index")
                    if embedding_model:
                        index, chunks, chunk_sources = create_document_index(
                            document_texts, 
                            embedding_model,
                            config.get("search.chunk_size", 1000),
                            config.get("search.chunk_overlap", 200)
                        )
                        st.session_state.document_index = index
                        st.session_state.document_chunks = chunks
                        st.session_state.chunk_sources = chunk_sources
                
                # Generate response
                if search_option == "Search specific document" and specific_doc:
                    app_logger.info(f"Searching specific document: {specific_doc}")
                    response, sources = generate_response(
                        user_question, 
                        document_texts, 
                        specific_doc, 
                        selected_model
                    )
                elif st.session_state.document_index is not None and embedding_model:
                    app_logger.info("Using semantic search across all documents")
                    response, sources = generate_response(
                        user_question,
                        document_texts,
                        None,
                        selected_model,
                        st.session_state.document_index,
                        st.session_state.document_chunks,
                        st.session_state.chunk_sources,
                        embedding_model
                    )
                else:
                    app_logger.info("Fallback to simple response generation")
                    response, sources = generate_response(user_question, document_texts, None, selected_model)
                
                # Store referenced documents
                st.session_state.referenced_docs = sources

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
            """, unsafe_allow_html=True)
            
            model_badge(selected_model)
            
            for source in sources:
                source_badge(source)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display the actual response
            response_card(response)

def insights_tab(document_texts: Optional[Dict[str, str]], selected_model: str) -> None:
    """
    Render the insights tab for document analysis and reporting.
    
    Args:
        document_texts: Dictionary of document texts
        selected_model: Selected Ollama model name
    """
    app_logger.info("Rendering insights tab")
    
    if not document_texts:
        st.info("Please upload documents in the Documents tab first.")
        return
        
    st.markdown("### 📊 Document Insights & Reports")
    
    st.markdown("""
    <div class="card">
        <p>Generate comprehensive reports and extract key insights from your documents.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get report types from config
    report_types = config.get("reports.report_types", [
        {"name": "Executive Summary", "prompt": "Create a concise executive summary of the document in 3-5 bullet points"},
        {"name": "Key Insights", "prompt": "Summarize the key insights and takeaways from this document"},
        {"name": "Full Analysis", "prompt": "Perform a detailed analysis of the document"}
    ])
    
    # Report generation options
    report_type = st.radio(
        "Report type:",
        [rt["name"] for rt in report_types],
        horizontal=True
    )
    
    # Get the prompt for the selected report type
    report_prompt = next((rt["prompt"] for rt in report_types if rt["name"] == report_type), 
                        "Summarize the key insights and takeaways from this document")
    
    if st.button("📄 Generate Report", use_container_width=True):
        with st.spinner(f"Generating {report_type.lower()}... This may take a minute."):
            app_logger.info(f"Generating {report_type} report")
            insights = {}
            
            for doc in document_texts:
                app_logger.info(f"Generating insights for document: {doc}")
                response, _ = generate_response(report_prompt, document_texts, doc, selected_model)
                insights[doc] = response
                
            # Generate PDF report
            app_logger.info("Generating PDF report")
            pdf_file = generate_pdf_report(
                insights, 
                title=f"{config.get('app.title', 'SentinelDocs')} - {report_type}"
            )
            
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
        except Exception as e:
            app_logger.error(f"Error removing temporary file: {str(e)}")

def main() -> None:
    """Main application entry point."""
    app_logger.info("Starting SentinelDocs application")
    
    # Initialize session state
    init_session_state()
    
    # Set up page configuration
    setup_page()
    
    # Check if Ollama is available
    ollama_available = check_ollama_availability()
    if not ollama_available:
        st.error("⚠️ Ollama service is not available. Please make sure the Ollama server is running locally.")
    
    # Get available models
    available_models = get_available_models()
    
    # Set up sidebar and get selected model
    selected_model = setup_sidebar(available_models)
    
    # Store selected model in session state for other components
    st.session_state.selected_model = selected_model
    
    # Load NLP model
    nlp = load_nlp_model()
    
    # Create main tabs for different functions
    main_tabs = st.tabs([
        create_tab_content("📂", "Documents"),
        create_tab_content("❓", "Ask Questions"),
        create_tab_content("📊", "Insights")
    ])
    
    # Document texts to share between tabs
    document_texts = None
    
    # Render document upload tab
    with main_tabs[0]:
        document_texts = documents_tab(document_texts, nlp)
        
    # Render questions tab
    with main_tabs[1]:
        questions_tab(document_texts, selected_model)
        
    # Render insights tab
    with main_tabs[2]:
        insights_tab(document_texts, selected_model)
    
    # Render footer
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
    
    app_logger.info("Application rendering complete")

if __name__ == "__main__":
    main() 