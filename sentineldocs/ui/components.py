"""
UI components for SentinelDocs.

This module provides reusable UI components and layouts for the Streamlit application.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Callable

def badge(text: str, color: str = "primary") -> None:
    """
    Display a badge with text.
    
    Args:
        text: The text to display in the badge
        color: The badge color (primary, secondary, success)
    """
    badge_class = f"badge badge-{color}"
    st.markdown(f'<span class="{badge_class}">{text}</span>', unsafe_allow_html=True)

def card(title: Optional[str] = None, content: Optional[str] = None) -> None:
    """
    Display a card with optional title and content.
    
    Args:
        title: Optional card title
        content: Optional card content as HTML
    """
    if title:
        st.markdown(f"""
        <div class="card">
            <h3>{title}</h3>
            {content or ""}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card">
            {content or ""}
        </div>
        """, unsafe_allow_html=True)

def document_status_card(document_name: str, stats: Dict[str, Any]) -> None:
    """
    Display a card with document statistics.
    
    Args:
        document_name: The name of the document
        stats: Dictionary of document statistics
    """
    st.markdown(f"""
    <div class="card">
        <h3>{document_name}</h3>
        <p><b>Words:</b> {stats.get('word_count', 'N/A')}</p>
        <p><b>Characters:</b> {stats.get('char_count', 'N/A')}</p>
        <p><b>Sentences:</b> {stats.get('sentences', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display entities if available
    if stats.get('entities'):
        st.markdown("<b>Key Entities:</b>", unsafe_allow_html=True)
        entity_cols = st.columns(4)
        
        entity_colors = {
            "PERSON": "#4F46E5",
            "ORG": "#06B6D4", 
            "GPE": "#10B981",
            "DATE": "#F59E0B",
            "MONEY": "#7C3AED",
            "TIME": "#EC4899"
        }
        
        col_idx = 0
        for entity_type, examples in stats['entities'].items():
            if examples:
                with entity_cols[col_idx % 4]:
                    for example in examples[:3]:  # Limit to 3 examples per type
                        color = entity_colors.get(entity_type, "#64748B")
                        st.markdown(f"""
                        <span style="
                            display: inline-block;
                            padding: 0.25rem 0.5rem;
                            border-radius: 9999px;
                            font-size: 0.75rem;
                            font-weight: 500;
                            margin-right: 0.5rem;
                            margin-bottom: 0.5rem;
                            background-color: {color};
                            color: white;">
                            {example}
                        </span>
                        """, unsafe_allow_html=True)
                col_idx += 1

def question_suggestions(questions: List[str], on_select: Callable[[str], None]) -> None:
    """
    Display a grid of question suggestion buttons.
    
    Args:
        questions: List of question suggestions
        on_select: Callback function when a question is selected
    """
    cols = st.columns(4)
    for i, question in enumerate(questions):
        col_idx = i % 4
        with cols[col_idx]:
            if st.button(question, key=f"q_{i}"):
                on_select(question)

def source_badge(source: str) -> None:
    """
    Display a badge for a document source.
    
    Args:
        source: Document source name
    """
    st.markdown(f"""
    <span class="badge badge-secondary">Source: {source}</span>
    """, unsafe_allow_html=True)

def model_badge(model: str) -> None:
    """
    Display a badge for the AI model used.
    
    Args:
        model: Model name
    """
    st.markdown(f"""
    <span class="badge badge-primary">Model: {model}</span>
    """, unsafe_allow_html=True)

def response_card(response: str) -> None:
    """
    Display an AI response in a card.
    
    Args:
        response: The AI response text
    """
    st.markdown(f"""
    <div class="card" style="background-color: #f8fafc;">
        {response.replace('\n', '<br>')}
    </div>
    """, unsafe_allow_html=True)

def create_tab_content(icon: str, title: str) -> str:
    """
    Create tab content with icon and title.
    
    Args:
        icon: Icon emoji
        title: Tab title
        
    Returns:
        Formatted tab title with icon
    """
    return f"{icon} {title}" 