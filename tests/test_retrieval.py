"""
Tests for Retrieval Module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLocalSearch:
    """Tests for LocalGraphSearch class."""
    
    def test_local_search_initialization(self):
        """Test local search initializes."""
        from src.retrieval.local_search import LocalGraphSearch
        
        search = LocalGraphSearch(
            entity_threshold=0.6,
            document_threshold=0.4,
            top_k=5
        )
        
        assert search.entity_threshold == 0.6
        assert search.top_k == 5
    
    def test_local_search_indexing(self):
        """Test indexing for local search."""
        from src.retrieval.local_search import LocalGraphSearch
        
        search = LocalGraphSearch()
        
        entities = [
            {'id': 'e1', 'normalized': 'ambedkar', 'text': 'Ambedkar'},
            {'id': 'e2', 'normalized': 'gandhi', 'text': 'Gandhi'}
        ]
        
        chunks = [
            {'id': 'c1', 'text': 'Ambedkar was a great leader.'},
            {'id': 'c2', 'text': 'Gandhi led the independence movement.'}
        ]
        
        # Mock embeddings
        entity_embeddings = np.random.rand(2, 384).astype(np.float32)
        chunk_embeddings = np.random.rand(2, 384).astype(np.float32)
        
        entity_to_chunks = {
            'ambedkar': ['c1'],
            'gandhi': ['c2']
        }
        
        search.index(
            entities=entities,
            chunks=chunks,
            entity_embeddings=entity_embeddings,
            chunk_embeddings=chunk_embeddings,
            entity_to_chunks=entity_to_chunks
        )
        
        assert len(search.entity_ids) == 2
        assert len(search.chunk_ids) == 2


class TestGlobalSearch:
    """Tests for GlobalGraphSearch class."""
    
    def test_global_search_initialization(self):
        """Test global search initializes."""
        from src.retrieval.global_search import GlobalGraphSearch
        
        search = GlobalGraphSearch(
            community_top_k=3,
            chunk_top_k=5
        )
        
        assert search.community_top_k == 3
        assert search.chunk_top_k == 5


class TestHybridSearch:
    """Tests for HybridSearch class."""
    
    def test_hybrid_search_initialization(self):
        """Test hybrid search initializes."""
        from src.retrieval.local_search import LocalGraphSearch
        from src.retrieval.global_search import GlobalGraphSearch
        from src.retrieval.hybrid_search import HybridSearch
        
        local = LocalGraphSearch()
        global_s = GlobalGraphSearch()
        
        hybrid = HybridSearch(
            local_search=local,
            global_search=global_s,
            local_weight=0.6,
            global_weight=0.4
        )
        
        assert hybrid.local_weight == 0.6
        assert hybrid.global_weight == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
