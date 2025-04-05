"""
Document processing module for SentinelDocs.

This module provides functions for extracting text from various document formats
and analyzing document content.
"""

import fitz  # PyMuPDF for PDFs
from docx import Document as DocxDocument
from typing import Dict, List, Optional, Any, Tuple
import spacy
from spacy.language import Language

# Type hints
DocumentText = Dict[str, str]
DocumentStats = Dict[str, Any]

# Load NLP model - this is now a function to handle errors gracefully
def load_nlp_model() -> Optional[Language]:
    """
    Load the spaCy NLP model for entity recognition.
    
    Returns:
        Optional[Language]: The loaded spaCy model or None if loading fails
    """
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Error loading NLP model: {str(e)}")
        return None

def extract_text_from_files(uploaded_files: List[Any]) -> DocumentText:
    """
    Extract text content from uploaded files of various formats.
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
        
    Returns:
        Dict[str, str]: Dictionary mapping filenames to extracted text
    """
    documents = {}
    
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split(".")[-1].lower()
        
        try:
            if file_type == "pdf":
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text = "\n".join([page.get_text() for page in doc])
            elif file_type == "docx":
                doc = DocxDocument(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif file_type == "txt":
                text = uploaded_file.read().decode("utf-8")
            else:
                print(f"Unsupported file format: {uploaded_file.name}")
                continue
                
            documents[uploaded_file.name] = text
        except Exception as e:
            print(f"Error processing {uploaded_file.name}: {str(e)}")
            
    return documents

def analyze_document_stats(documents: DocumentText, nlp: Optional[Language] = None) -> DocumentStats:
    """
    Generate statistical analysis of document content.
    
    Args:
        documents: Dictionary mapping filenames to document text
        nlp: Optional spaCy language model for NLP tasks
        
    Returns:
        Dict[str, Any]: Dictionary containing document statistics
    """
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
                # Limit to first 100k chars for performance
                doc = nlp(text[:100000])  
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
                print(f"Error analyzing entities in {doc_name}: {str(e)}")
        
        stats[doc_name] = doc_stats
    
    return stats 