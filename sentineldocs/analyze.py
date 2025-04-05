"""
Analysis module for SentinelDocs.

This module provides functions for semantic search and analysis of documents 
using embeddings and language models.
"""

from typing import Dict, List, Any, Optional, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Custom types for clarity
DocumentText = Dict[str, str]
DocumentIndex = Any  # FAISS index
EmbeddingModel = Any  # Sentence transformer model
SearchResult = Dict[str, Any]
SearchResults = List[SearchResult]
ComparisonResults = Dict[str, str]

def load_embedding_model() -> Optional[SentenceTransformer]:
    """
    Load the embedding model for semantic search.
    
    Returns:
        Optional[SentenceTransformer]: The embedding model or None if loading fails
    """
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        return None

def create_document_index(
    documents: DocumentText, 
    embedding_model: SentenceTransformer,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[DocumentIndex, List[str], List[str]]:
    """
    Create a searchable index from document text.
    
    Args:
        documents: Dictionary mapping filenames to document text
        embedding_model: Model to create embeddings
        chunk_size: Size of text chunks for indexing
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        Tuple containing:
            - FAISS index
            - List of text chunks
            - List of source document names for each chunk
    """
    chunks = []
    chunk_sources = []
    
    # Create chunks from documents
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

def search_documents(
    query: str, 
    index: DocumentIndex, 
    chunks: List[str], 
    chunk_sources: List[str], 
    embedding_model: SentenceTransformer,
    top_k: int = 3
) -> SearchResults:
    """
    Search for relevant document chunks using semantic search.
    
    Args:
        query: Search query
        index: FAISS index of document chunks
        chunks: List of document chunks
        chunk_sources: List of source document names for each chunk
        embedding_model: Model to create query embedding
        top_k: Number of results to return
        
    Returns:
        List of search results containing text, source, and score
    """
    try:
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
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return []

def generate_response(
    question: str, 
    documents: DocumentText, 
    doc_name: Optional[str] = None, 
    model_name: str = "deepseek-r1:8b",
    index: Optional[DocumentIndex] = None,
    chunks: Optional[List[str]] = None,
    chunk_sources: Optional[List[str]] = None,
    embedding_model: Optional[SentenceTransformer] = None
) -> Tuple[str, List[str]]:
    """
    Generate a response to a question about the documents.
    
    Args:
        question: User question
        documents: Dictionary mapping filenames to document text
        doc_name: Optional specific document to search
        model_name: Name of the Ollama model to use
        index: Optional FAISS index for semantic search
        chunks: Optional list of document chunks
        chunk_sources: Optional list of chunk sources
        embedding_model: Optional embedding model
        
    Returns:
        Tuple containing:
            - Response text
            - List of source documents used
    """
    try:
        # If a specific document is provided, use it directly
        if doc_name and doc_name in documents:
            document_text = documents[doc_name]
            context = document_text[:5000]  # Use first 5000 chars
            referenced_docs = [doc_name]
        # Otherwise use semantic search to find relevant parts
        elif index is not None and chunks is not None and chunk_sources is not None and embedding_model is not None:
            # Search for relevant chunks
            results = search_documents(question, index, chunks, chunk_sources, embedding_model)
            context = "\n\n".join([r["text"] for r in results])
            
            # For traceability, store which documents were referenced
            referenced_docs = list(set([r["source"] for r in results]))
        else:
            # Fallback to just using the first document
            if documents:
                doc_name = list(documents.keys())[0]
                context = documents[doc_name][:5000]
                referenced_docs = [doc_name]
            else:
                return "No documents available", []
        
        # Generate response
        llm = OllamaLLM(model=model_name)
        prompt = PromptTemplate.from_template(
            """
            You are an AI assistant analyzing documents. Given the extracted text:
            {document_text}
            
            Answer concisely and accurately:
            {question}
            """
        )
        response = llm.invoke(prompt.format(document_text=context, question=question))
        return response, referenced_docs
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}", []

def compare_documents(
    documents: DocumentText, 
    model_name: str = "deepseek-r1:8b"
) -> ComparisonResults:
    """
    Compare multiple documents to find similarities and differences.
    
    Args:
        documents: Dictionary mapping filenames to document text
        model_name: Name of the Ollama model to use
        
    Returns:
        Dictionary mapping pairs of documents to comparison results
    """
    # Require at least 2 documents
    if len(documents) < 2:
        return {"error": "Need at least 2 documents to compare"}
    
    try:
        llm = OllamaLLM(model=model_name)
        results = {}
        
        # Compare each pair of documents
        doc_names = list(documents.keys())
        for i in range(len(doc_names)):
            for j in range(i+1, len(doc_names)):
                doc1 = doc_names[i]
                doc2 = doc_names[j]
                
                # Truncate documents if they're too long
                text1 = documents[doc1][:5000]
                text2 = documents[doc2][:5000]
                
                prompt = PromptTemplate.from_template(
                    """
                    You are a document comparison expert. Compare the following two documents and identify key similarities and differences.
                    
                    DOCUMENT 1: {doc1_name}
                    {doc1_text}
                    
                    DOCUMENT 2: {doc2_name}
                    {doc2_text}
                    
                    Provide a concise comparison between the two documents, highlighting:
                    1. Key similarities in content and themes
                    2. Major differences in content, structure, or focus
                    3. A brief summary of what each document uniquely contributes
                    """
                )
                
                comparison = llm.invoke(prompt.format(
                    doc1_name=doc1,
                    doc1_text=text1,
                    doc2_name=doc2,
                    doc2_text=text2
                ))
                
                results[f"{doc1} vs {doc2}"] = comparison
        
        return results
    except Exception as e:
        logger.error(f"Error comparing documents: {str(e)}")
        return {"error": f"Error comparing documents: {str(e)}"} 