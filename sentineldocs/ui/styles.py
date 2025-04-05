"""
UI styles and CSS for SentinelDocs.

This module provides CSS styles and visual theming for the application.
"""

# Main CSS styles for the application
MAIN_CSS = """
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
"""

# Header with logo 
HEADER_HTML = """
<div class="app-header">
    <div class="logo-img" style="font-size: 80px; display: flex; justify-content: center; align-items: center;">
        📄
    </div>
    <h1>SentinelDocs</h1>
    <p style="color: var(--text-light); font-size: 1.2rem; margin-top: -0.5rem;">Your Private AI-Powered Document Analyst</p>
</div>
"""

# Footer with additional information
FOOTER_HTML = """
<div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0;">
    <p style="color: #64748b; font-size: 0.9rem;">
        SentinelDocs - Your documents never leave your machine. All processing happens locally.
    </p>
</div>
"""

# Card template
def create_card(content: str) -> str:
    """
    Create a styled card with the given content.
    
    Args:
        content: HTML content for the card
        
    Returns:
        HTML string for a styled card
    """
    return f'<div class="card">{content}</div>' 