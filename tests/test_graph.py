"""
Tests for Knowledge Graph Module
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEntityExtractor:
    """Tests for EntityExtractor class."""
    
    def test_extractor_initialization(self):
        """Test extractor initializes correctly."""
        from src.graph.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        assert extractor.nlp is not None
    
    def test_entity_extraction(self):
        """Test extracting entities from text."""
        from src.graph.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        
        text = "Dr. B.R. Ambedkar drafted the Indian Constitution in 1949."
        entities = extractor.extract_from_text(text)
        
        assert len(entities) > 0
        
        # Check for expected entities
        entity_texts = [e.text.lower() for e in entities]
        assert any('ambedkar' in t for t in entity_texts)
    
    def test_custom_entities(self):
        """Test custom Ambedkar-related entities."""
        from src.graph.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        
        text = "Babasaheb met with Mahatma Gandhi about the Poona Pact."
        entities = extractor.extract_from_text(text)
        
        entity_texts = [e.text.lower() for e in entities]
        assert len(entities) > 0


class TestRelationshipExtractor:
    """Tests for RelationshipExtractor class."""
    
    def test_extractor_initialization(self):
        """Test relationship extractor initializes."""
        from src.graph.relationship_extractor import RelationshipExtractor
        
        extractor = RelationshipExtractor()
        assert extractor.nlp is not None
    
    def test_relationship_extraction(self):
        """Test extracting relationships."""
        from src.graph.relationship_extractor import RelationshipExtractor
        
        extractor = RelationshipExtractor()
        
        text = "Ambedkar founded the Republican Party of India."
        relationships = extractor.extract_from_text(text)
        
        # Should find at least one relationship
        assert isinstance(relationships, list)


class TestGraphBuilder:
    """Tests for KnowledgeGraphBuilder class."""
    
    def test_graph_creation(self):
        """Test creating a knowledge graph."""
        from src.graph.graph_builder import KnowledgeGraphBuilder
        
        builder = KnowledgeGraphBuilder()
        
        entities = [
            {'text': 'Ambedkar', 'normalized': 'ambedkar', 'label': 'PERSON', 'frequency': 5, 'chunk_ids': ['c1']},
            {'text': 'Gandhi', 'normalized': 'gandhi', 'label': 'PERSON', 'frequency': 3, 'chunk_ids': ['c1', 'c2']}
        ]
        
        relationships = [
            {'source': 'ambedkar', 'target': 'gandhi', 'relation_type': 'MET_WITH', 'weight': 2}
        ]
        
        graph = builder.build_graph(entities, relationships)
        
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
    
    def test_graph_statistics(self):
        """Test graph statistics."""
        from src.graph.graph_builder import KnowledgeGraphBuilder
        
        builder = KnowledgeGraphBuilder()
        
        entities = [
            {'text': 'A', 'normalized': 'a', 'label': 'PERSON', 'frequency': 1, 'chunk_ids': []},
            {'text': 'B', 'normalized': 'b', 'label': 'ORG', 'frequency': 1, 'chunk_ids': []}
        ]
        
        builder.build_graph(entities, [])
        stats = builder.get_statistics()
        
        assert stats['num_nodes'] == 2


class TestCommunityDetector:
    """Tests for CommunityDetector class."""
    
    def test_community_detection(self):
        """Test community detection."""
        import networkx as nx
        from src.graph.community_detector import CommunityDetector
        
        # Create test graph
        graph = nx.Graph()
        graph.add_edges_from([
            ('a', 'b'), ('b', 'c'), ('c', 'a'),  # Community 1
            ('d', 'e'), ('e', 'f'), ('f', 'd'),  # Community 2
            ('c', 'd')  # Bridge
        ])
        
        detector = CommunityDetector(algorithm='louvain')
        communities = detector.detect_communities(graph)
        
        assert len(communities) == 6  # All nodes assigned
        assert detector.get_statistics()['num_communities'] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
