"""
Tests for Semantic Chunking Module
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSemanticChunker:
    """Tests for SemanticChunker class."""
    
    def test_chunker_initialization(self):
        """Test chunker initializes correctly."""
        from src.chunking.semantic_chunker import SemanticChunker
        
        chunker = SemanticChunker(
            similarity_threshold=0.5,
            max_tokens=1024
        )
        
        assert chunker.similarity_threshold == 0.5
        assert chunker.max_tokens == 1024
        assert chunker.model is not None
    
    def test_basic_chunking(self):
        """Test basic text chunking."""
        from src.chunking.semantic_chunker import SemanticChunker
        
        chunker = SemanticChunker(similarity_threshold=0.3)
        
        text = """
        Dr. B.R. Ambedkar was a social reformer. He fought against caste discrimination.
        He drafted the Indian Constitution. He believed in education and equality.
        The weather today is sunny. It is a good day for a walk.
        """
        
        chunks = chunker.chunk(text, doc_id="test")
        
        assert len(chunks) > 0
        assert all(c.text for c in chunks)
        assert all(c.id.startswith("test_chunk_") for c in chunks)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        from src.chunking.semantic_chunker import SemanticChunker
        
        chunker = SemanticChunker()
        chunks = chunker.chunk("", doc_id="empty")
        
        assert len(chunks) == 0
    
    def test_single_sentence(self):
        """Test chunking single sentence."""
        from src.chunking.semantic_chunker import SemanticChunker
        
        chunker = SemanticChunker()
        chunks = chunker.chunk("This is a single sentence.", doc_id="single")
        
        assert len(chunks) == 1
    
    def test_chunk_has_embedding(self):
        """Test chunks have embeddings."""
        from src.chunking.semantic_chunker import SemanticChunker
        
        chunker = SemanticChunker()
        text = "First sentence about Ambedkar. Second sentence about constitution."
        chunks = chunker.chunk(text, doc_id="embed_test")
        
        assert all(c.embedding is not None for c in chunks)


class TestPDFLoader:
    """Tests for PDF loading."""
    
    def test_pdf_extraction(self):
        """Test PDF text extraction."""
        from src.chunking.pdf_loader import extract_text_from_pdf
        
        pdf_path = Path(__file__).parent.parent / "data" / "Ambedkar_book.pdf"
        
        if not pdf_path.exists():
            pytest.skip("Test PDF not found")
        
        text, metadata = extract_text_from_pdf(str(pdf_path))
        
        assert len(text) > 0
        assert metadata.get('pages', 0) > 0
    
    def test_text_cleaning(self):
        """Test text cleaning function."""
        from src.chunking.pdf_loader import clean_text
        
        dirty_text = "  Too   many    spaces   \n\n\n  Page 123  "
        clean = clean_text(dirty_text)
        
        assert "  " not in clean
        assert "Page 123" not in clean


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
