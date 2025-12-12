"""
PDF Loader - Extract text from PDF files

Handles text extraction from PDF documents for the SemRAG pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, Dict]:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (extracted_text, metadata)
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Extracting text from: {pdf_path}")
    
    text = ""
    metadata = {
        "filename": pdf_path.name,
        "pages": 0,
        "extraction_method": None
    }
    
    # Try pdfplumber first (better for complex layouts)
    try:
        import pdfplumber
        
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            
            text = "\n\n".join(pages_text)
            metadata["pages"] = len(pdf.pages)
            metadata["extraction_method"] = "pdfplumber"
            
        logger.info(f"Extracted {len(text)} characters using pdfplumber")
        
    except ImportError:
        logger.warning("pdfplumber not available, trying pypdf")
        
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(pdf_path)
            pages_text = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            
            text = "\n\n".join(pages_text)
            metadata["pages"] = len(reader.pages)
            metadata["extraction_method"] = "pypdf"
            
            logger.info(f"Extracted {len(text)} characters using pypdf")
            
        except ImportError:
            raise ImportError("Neither pdfplumber nor pypdf is installed. "
                            "Please install: pip install pdfplumber pypdf")
    
    # Clean up the text
    text = clean_text(text)
    
    return text, metadata


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing artifacts and normalizing whitespace.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers (common patterns)
    text = re.sub(r'\b\d+\s*\|\s*Page\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
    
    # Remove very short lines (likely headers/footers)
    lines = text.split('\n')
    lines = [l for l in lines if len(l.strip()) > 20 or l.strip() == '']
    text = '\n'.join(lines)
    
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    return text.strip()


def extract_with_page_info(pdf_path: str) -> List[Dict]:
    """
    Extract text with page number information.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of dicts with 'page', 'text' keys
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    pages_data = []
    
    try:
        import pdfplumber
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    pages_data.append({
                        'page': i + 1,
                        'text': clean_text(page_text)
                    })
                    
    except ImportError:
        from pypdf import PdfReader
        
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                pages_data.append({
                    'page': i + 1,
                    'text': clean_text(page_text)
                })
    
    logger.info(f"Extracted {len(pages_data)} pages with text")
    return pages_data
