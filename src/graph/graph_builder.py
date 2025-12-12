"""
Graph Builder - Construct knowledge graph from entities and relationships

Creates a NetworkX graph with:
- Nodes: Entities with attributes
- Edges: Relationships with weights
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import pickle
import json

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Builds and manages a knowledge graph for the SemRAG system.
    
    The graph structure:
    - Nodes represent entities with embeddings
    - Edges represent relationships with weights
    - Nodes are linked to source chunks
    """
    
    def __init__(self):
        """Initialize the knowledge graph builder."""
        self.graph = nx.Graph()
        self.entity_to_node = {}  # Map entity text to node ID
        self.chunk_to_entities = {}  # Map chunk ID to entities
        self.entity_embeddings = {}  # Entity embeddings
        
    def build_graph(
        self,
        entities: List[Dict],
        relationships: List[Dict],
        chunks: Optional[List[Dict]] = None
    ) -> nx.Graph:
        """
        Build the knowledge graph from entities and relationships.
        
        Args:
            entities: List of entity dicts
            relationships: List of relationship dicts
            chunks: Optional list of chunks for linking
            
        Returns:
            NetworkX graph
        """
        logger.info("Building knowledge graph...")
        
        # Clear existing graph
        self.graph = nx.Graph()
        self.entity_to_node = {}
        
        # Add entity nodes
        for i, entity in enumerate(entities):
            node_id = f"entity_{i}"
            normalized = entity.get('normalized', entity.get('text', ''))
            
            self.graph.add_node(
                node_id,
                text=entity.get('text', normalized),
                normalized=normalized,
                label=entity.get('label', 'UNKNOWN'),
                frequency=entity.get('frequency', 1),
                chunk_ids=entity.get('chunk_ids', []),
                node_type='entity'
            )
            
            self.entity_to_node[normalized.lower()] = node_id
        
        logger.info(f"Added {len(entities)} entity nodes")
        
        # Add relationship edges
        edges_added = 0
        for rel in relationships:
            source = rel.get('source', '').lower()
            target = rel.get('target', '').lower()
            
            source_node = self.entity_to_node.get(source)
            target_node = self.entity_to_node.get(target)
            
            if source_node and target_node and source_node != target_node:
                # Add or update edge
                if self.graph.has_edge(source_node, target_node):
                    # Increment weight if edge exists
                    self.graph[source_node][target_node]['weight'] += rel.get('weight', 1)
                    # Add relationship type if new
                    existing_types = self.graph[source_node][target_node].get('relation_types', [])
                    new_type = rel.get('relation_type', 'RELATED')
                    if new_type not in existing_types:
                        existing_types.append(new_type)
                        self.graph[source_node][target_node]['relation_types'] = existing_types
                else:
                    self.graph.add_edge(
                        source_node,
                        target_node,
                        weight=rel.get('weight', 1),
                        relation_types=[rel.get('relation_type', 'RELATED')],
                        chunk_ids=rel.get('chunk_ids', [])
                    )
                    edges_added += 1
        
        logger.info(f"Added {edges_added} relationship edges")
        
        # Build chunk-to-entity mapping
        if chunks:
            self._build_chunk_mapping(chunks, entities)
        
        logger.info(f"Knowledge graph built: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _build_chunk_mapping(self, chunks: List[Dict], entities: List[Dict]):
        """Build mapping from chunks to entities."""
        self.chunk_to_entities = {}
        
        for entity in entities:
            for chunk_id in entity.get('chunk_ids', []):
                if chunk_id not in self.chunk_to_entities:
                    self.chunk_to_entities[chunk_id] = []
                
                normalized = entity.get('normalized', entity.get('text', ''))
                if normalized not in self.chunk_to_entities[chunk_id]:
                    self.chunk_to_entities[chunk_id].append(normalized)
    
    def add_embeddings(
        self,
        embeddings: Dict[str, np.ndarray]
    ):
        """
        Add embeddings to entity nodes.
        
        Args:
            embeddings: Dict mapping normalized entity text to embedding
        """
        for entity_text, embedding in embeddings.items():
            node_id = self.entity_to_node.get(entity_text.lower())
            if node_id:
                self.entity_embeddings[node_id] = embedding
        
        logger.info(f"Added embeddings for {len(self.entity_embeddings)} entities")
    
    def get_neighbors(
        self,
        entity_text: str,
        max_depth: int = 1
    ) -> List[Dict]:
        """
        Get neighboring entities in the graph.
        
        Args:
            entity_text: Entity to find neighbors for
            max_depth: Maximum hop distance
            
        Returns:
            List of neighbor entity dicts
        """
        node_id = self.entity_to_node.get(entity_text.lower())
        if not node_id:
            return []
        
        neighbors = []
        visited = {node_id}
        current_level = [node_id]
        
        for depth in range(max_depth):
            next_level = []
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
                        
                        node_data = self.graph.nodes[neighbor]
                        edge_data = self.graph[node][neighbor]
                        
                        neighbors.append({
                            'node_id': neighbor,
                            'text': node_data.get('text', ''),
                            'label': node_data.get('label', ''),
                            'distance': depth + 1,
                            'relation_types': edge_data.get('relation_types', []),
                            'edge_weight': edge_data.get('weight', 1)
                        })
            
            current_level = next_level
        
        return neighbors
    
    def get_entities_for_chunk(self, chunk_id: str) -> List[str]:
        """Get entities mentioned in a specific chunk."""
        return self.chunk_to_entities.get(chunk_id, [])
    
    def get_chunks_for_entity(self, entity_text: str) -> List[str]:
        """Get chunks that mention a specific entity."""
        node_id = self.entity_to_node.get(entity_text.lower())
        if not node_id:
            return []
        
        return self.graph.nodes[node_id].get('chunk_ids', [])
    
    def get_subgraph(
        self,
        entity_texts: List[str],
        include_neighbors: bool = True
    ) -> nx.Graph:
        """
        Extract a subgraph containing specified entities.
        
        Args:
            entity_texts: List of entity texts to include
            include_neighbors: Whether to include direct neighbors
            
        Returns:
            NetworkX subgraph
        """
        node_ids = set()
        
        for text in entity_texts:
            node_id = self.entity_to_node.get(text.lower())
            if node_id:
                node_ids.add(node_id)
                
                if include_neighbors:
                    for neighbor in self.graph.neighbors(node_id):
                        node_ids.add(neighbor)
        
        return self.graph.subgraph(node_ids).copy()
    
    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
            'num_connected_components': nx.number_connected_components(self.graph),
            'nodes_by_type': {},
            'avg_degree': 0
        }
        
        # Count by entity type
        for node, data in self.graph.nodes(data=True):
            label = data.get('label', 'UNKNOWN')
            stats['nodes_by_type'][label] = stats['nodes_by_type'].get(label, 0) + 1
        
        # Average degree
        if self.graph.number_of_nodes() > 0:
            stats['avg_degree'] = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        
        return stats
    
    def save(self, path: str):
        """
        Save the knowledge graph to disk.
        
        Args:
            path: Path to save to (pickle format)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'graph': self.graph,
            'entity_to_node': self.entity_to_node,
            'chunk_to_entities': self.chunk_to_entities,
            'entity_embeddings': self.entity_embeddings
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Knowledge graph saved to {path}")
    
    def load(self, path: str):
        """
        Load the knowledge graph from disk.
        
        Args:
            path: Path to load from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.graph = data['graph']
        self.entity_to_node = data['entity_to_node']
        self.chunk_to_entities = data.get('chunk_to_entities', {})
        self.entity_embeddings = data.get('entity_embeddings', {})
        
        logger.info(f"Knowledge graph loaded from {path}: "
                   f"{self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
    
    def export_for_visualization(self) -> Dict:
        """
        Export graph data for visualization.
        
        Returns:
            Dict with nodes and edges for visualization
        """
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'label': data.get('text', node_id),
                'type': data.get('label', 'UNKNOWN'),
                'size': min(data.get('frequency', 1) * 5 + 10, 50)
            })
        
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                'from': source,
                'to': target,
                'weight': data.get('weight', 1),
                'label': ', '.join(data.get('relation_types', [])[:1])
            })
        
        return {'nodes': nodes, 'edges': edges}
