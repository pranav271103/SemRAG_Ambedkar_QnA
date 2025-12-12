"""
Community Detector - Detect communities in knowledge graph

Uses Louvain or Leiden algorithm for community detection
to group related entities together.
"""

import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)


class CommunityDetector:
    """
    Detects communities in the knowledge graph using
    Louvain or Leiden algorithm.
    
    Communities group thematically related entities together,
    which is used for global search in SemRAG.
    """
    
    def __init__(self, algorithm: str = "louvain", resolution: float = 1.0):
        """
        Initialize community detector.
        
        Args:
            algorithm: "louvain" or "leiden"
            resolution: Resolution parameter (higher = more communities)
        """
        self.algorithm = algorithm.lower()
        self.resolution = resolution
        self.communities = {}  # node_id -> community_id
        self.community_nodes = defaultdict(list)  # community_id -> [node_ids]
        
    def detect_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """
        Detect communities in the graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dict mapping node IDs to community IDs
        """
        if graph.number_of_nodes() == 0:
            logger.warning("Empty graph, no communities to detect")
            return {}
        
        logger.info(f"Detecting communities using {self.algorithm} algorithm...")
        
        if self.algorithm == "louvain":
            self.communities = self._louvain_detection(graph)
        elif self.algorithm == "leiden":
            self.communities = self._leiden_detection(graph)
        else:
            logger.warning(f"Unknown algorithm {self.algorithm}, using Louvain")
            self.communities = self._louvain_detection(graph)
        
        # Build community to nodes mapping
        self.community_nodes = defaultdict(list)
        for node_id, comm_id in self.communities.items():
            self.community_nodes[comm_id].append(node_id)
        
        num_communities = len(self.community_nodes)
        logger.info(f"Detected {num_communities} communities")
        
        # Log community sizes
        sizes = [len(nodes) for nodes in self.community_nodes.values()]
        if sizes:
            logger.info(f"Community sizes - Min: {min(sizes)}, Max: {max(sizes)}, "
                       f"Avg: {sum(sizes)/len(sizes):.1f}")
        
        return self.communities
    
    def _louvain_detection(self, graph: nx.Graph) -> Dict[str, int]:
        """Use Louvain algorithm for community detection."""
        try:
            import community as community_louvain
            
            # Louvain needs connected graph components
            partition = community_louvain.best_partition(
                graph,
                resolution=self.resolution,
                random_state=42
            )
            return partition
            
        except ImportError:
            logger.warning("python-louvain not installed, using NetworkX fallback")
            return self._networkx_fallback(graph)
    
    def _leiden_detection(self, graph: nx.Graph) -> Dict[str, int]:
        """Use Leiden algorithm for community detection."""
        try:
            import leidenalg
            import igraph as ig
            
            # Convert NetworkX to igraph
            edges = list(graph.edges())
            ig_graph = ig.Graph.TupleList(edges, directed=False)
            
            # Add node names
            node_list = list(graph.nodes())
            ig_graph.vs['name'] = node_list
            
            # Get edge weights if available
            weights = [graph[u][v].get('weight', 1.0) for u, v in edges]
            
            # Run Leiden
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                weights=weights,
                seed=42
            )
            
            # Map back to NetworkX node IDs
            result = {}
            for i, comm in enumerate(partition):
                for node_idx in comm:
                    node_name = ig_graph.vs[node_idx]['name']
                    result[node_name] = i
            
            return result
            
        except ImportError:
            logger.warning("leidenalg not installed, falling back to Louvain")
            return self._louvain_detection(graph)
    
    def _networkx_fallback(self, graph: nx.Graph) -> Dict[str, int]:
        """Fallback community detection using NetworkX."""
        try:
            # Use greedy modularity communities
            from networkx.algorithms.community import greedy_modularity_communities
            
            communities = list(greedy_modularity_communities(graph))
            
            result = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    result[node] = comm_id
            
            return result
            
        except Exception as e:
            logger.warning(f"Fallback failed: {e}, assigning single community")
            return {node: 0 for node in graph.nodes()}
    
    def get_community(self, node_id: str) -> Optional[int]:
        """Get community ID for a node."""
        return self.communities.get(node_id)
    
    def get_community_nodes(self, community_id: int) -> List[str]:
        """Get all nodes in a community."""
        return self.community_nodes.get(community_id, [])
    
    def get_community_subgraph(
        self,
        graph: nx.Graph,
        community_id: int
    ) -> nx.Graph:
        """
        Extract subgraph for a specific community.
        
        Args:
            graph: Full graph
            community_id: Community to extract
            
        Returns:
            Subgraph containing only nodes in the community
        """
        nodes = self.community_nodes.get(community_id, [])
        return graph.subgraph(nodes).copy()
    
    def get_community_info(
        self,
        graph: nx.Graph,
        community_id: int
    ) -> Dict:
        """
        Get information about a community.
        
        Args:
            graph: Full graph
            community_id: Community to examine
            
        Returns:
            Dict with community information
        """
        nodes = self.community_nodes.get(community_id, [])
        
        if not nodes:
            return {'id': community_id, 'size': 0, 'entities': []}
        
        entities = []
        for node_id in nodes:
            node_data = graph.nodes[node_id]
            entities.append({
                'node_id': node_id,
                'text': node_data.get('text', node_id),
                'label': node_data.get('label', 'UNKNOWN'),
                'frequency': node_data.get('frequency', 1)
            })
        
        # Sort by frequency
        entities.sort(key=lambda x: x['frequency'], reverse=True)
        
        # Get internal edges
        subgraph = graph.subgraph(nodes)
        
        return {
            'id': community_id,
            'size': len(nodes),
            'entities': entities,
            'internal_edges': subgraph.number_of_edges(),
            'top_entities': [e['text'] for e in entities[:5]]
        }
    
    def get_all_communities_info(self, graph: nx.Graph) -> List[Dict]:
        """Get information about all communities."""
        communities_info = []
        
        for comm_id in sorted(self.community_nodes.keys()):
            info = self.get_community_info(graph, comm_id)
            communities_info.append(info)
        
        # Sort by size
        communities_info.sort(key=lambda x: x['size'], reverse=True)
        
        return communities_info
    
    def get_statistics(self) -> Dict:
        """Get community detection statistics."""
        if not self.community_nodes:
            return {'num_communities': 0}
        
        sizes = [len(nodes) for nodes in self.community_nodes.values()]
        
        return {
            'num_communities': len(self.community_nodes),
            'total_nodes': sum(sizes),
            'min_size': min(sizes),
            'max_size': max(sizes),
            'avg_size': sum(sizes) / len(sizes),
            'size_distribution': {
                'small (1-3)': sum(1 for s in sizes if s <= 3),
                'medium (4-10)': sum(1 for s in sizes if 4 <= s <= 10),
                'large (>10)': sum(1 for s in sizes if s > 10)
            }
        }
