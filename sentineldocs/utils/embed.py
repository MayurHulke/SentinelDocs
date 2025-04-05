"""
Embedding utilities for SentinelDocs.

This module provides functions for working with document embeddings and vectors.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from .logging import get_logger

# Get logger for this module
logger = get_logger("utils.embed")

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
    """
    Load the embedding model for semantic search.
    
    Args:
        model_name: Name of the sentence-transformers model to load
        
    Returns:
        Optional[SentenceTransformer]: The embedding model or None if loading fails
    """
    try:
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        return None

def create_embeddings(
    texts: List[str], 
    model: SentenceTransformer
) -> np.ndarray:
    """
    Create embeddings for a list of text chunks.
    
    Args:
        texts: List of text chunks to embed
        model: SentenceTransformer model for creating embeddings
        
    Returns:
        numpy.ndarray containing the embeddings
    """
    try:
        logger.info(f"Creating embeddings for {len(texts)} text chunks")
        embeddings = model.encode(texts)
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        # Return empty array with correct dimensions as fallback
        if hasattr(model, 'get_sentence_embedding_dimension'):
            dim = model.get_sentence_embedding_dimension()
        else:
            dim = 384  # Default for many sentence-transformers models
        return np.zeros((len(texts), dim))

def create_faiss_index(embeddings: np.ndarray) -> Optional[faiss.Index]:
    """
    Create a FAISS index from embeddings.
    
    Args:
        embeddings: numpy.ndarray containing embeddings
        
    Returns:
        FAISS index or None if creation fails
    """
    try:
        logger.info(f"Creating FAISS index with {len(embeddings)} vectors")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        logger.info(f"FAISS index created successfully with {index.ntotal} vectors")
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        return None

def search_index(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    texts: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks in the index.
    
    Args:
        query: Search query
        model: SentenceTransformer model for encoding query
        index: FAISS index to search
        texts: Original text chunks corresponding to index vectors
        metadata: Optional metadata for each text chunk
        top_k: Number of results to return
        
    Returns:
        List of dictionaries containing search results
    """
    try:
        logger.info(f"Searching index for query: {query}")
        query_embedding = model.encode([query])
        distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(texts):  # Ensure index is valid
                result = {
                    "text": texts[idx],
                    "score": float(distances[0][i])
                }
                
                # Add metadata if available
                if metadata and idx < len(metadata):
                    for k, v in metadata[idx].items():
                        result[k] = v
                        
                results.append(result)
                
        logger.info(f"Search returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error searching index: {str(e)}")
        return [] 