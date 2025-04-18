"""
PDF generation utilities for SentinelDocs.

This module provides functionality for creating PDF reports based on document analysis.
"""

import os
import tempfile
from typing import Dict, Optional
from fpdf import FPDF
from datetime import datetime

class DocumentReport(FPDF):
    """PDF report generator for document insights."""
    
    def __init__(self, title: str = "Document Analysis Report"):
        """
        Initialize a new PDF report.
        
        Args:
            title: The title of the report
        """
        super().__init__()
        self.title = title
        self.set_author("SentinelDocs")
        self.set_creator("SentinelDocs")
        
    def header(self):
        """Add a header to each page of the report."""
        # Set font
        self.set_font("Arial", "B", 12)
        
        # Add logo or title
        self.cell(0, 10, self.title, 0, 1, "C")
        
        # Add date
        self.set_font("Arial", "I", 8)
        self.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, "R")
        
        # Add line
        self.line(10, 25, 200, 25)
        self.ln(10)
        
    def footer(self):
        """Add a footer to each page of the report."""
        # Set position at 1.5 cm from bottom
        self.set_y(-15)
        
        # Set font
        self.set_font("Arial", "I", 8)
        
        # Add page number
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")
        
    def add_section(self, title: str):
        """
        Add a section heading to the report.
        
        Args:
            title: The section title
        """
        self.set_font("Arial", "B", 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, 0, 1, "L", True)
        self.ln(5)
        
    def add_document_insight(self, doc_name: str, insight: str):
        """
        Add document insight to the report.
        
        Args:
            doc_name: The name of the document
            insight: The insight text
        """
        self.set_font("Arial", "B", 11)
        self.cell(0, 10, f"Document: {doc_name}", 0, 1)
        
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 5, insight)
        self.ln(5)

def generate_pdf_report(
    insights: Dict[str, str], 
    title: str = "SentinelDocs - Document Insights Report",
    output_dir: Optional[str] = None
) -> str:
    """
    Generate a PDF report containing document insights.
    
    Args:
        insights: Dictionary mapping document names to insight text
        title: Report title
        output_dir: Optional directory to save the report (uses temp dir if None)
        
    Returns:
        The path to the generated PDF file
    """
    pdf = DocumentReport(title)
    pdf.add_page()
    
    # Add introduction
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 5, 
        "This report contains AI-generated insights from the analyzed documents. "
        "All processing was performed locally with no data sent to external services."
    )
    pdf.ln(5)
    
    # Add document insights
    pdf.add_section("Document Insights")
    
    for doc_name, insight in insights.items():
        pdf.add_document_insight(doc_name, insight)
        
    # Add metadata section
    pdf.add_section("Analysis Metadata")
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, 
        f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Number of Documents: {len(insights)}\n"
        f"Generated by: SentinelDocs"
    )
    
    # Save the PDF
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"SentinelDocs_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    else:
        output_path = tempfile.mktemp(suffix=".pdf", prefix="SentinelDocs_Report_")
        
    pdf.output(output_path)
    return output_path 