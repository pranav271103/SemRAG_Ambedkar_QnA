"""
Query Engine - Main interface for querying the SemRAG system

Loads indices and provides a unified interface for:
- Local search
- Global search
- Hybrid search
- Answer generation
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Generator
import yaml
import numpy as np

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Main query interface for the SemRAG system.
    
    Loads pre-built indices and provides methods for
    searching and generating answers.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize query engine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Data stores
        self.chunks = []
        self.entities = []
        self.communities_info = []
        self.community_summaries = {}
        self.chunk_embeddings = None
        self.entity_embeddings = None
        self.entity_to_chunks = {}
        self.community_to_chunks = {}
        
        # Components (lazy loaded)
        self._embedding_model = None
        self._graph_builder = None
        self._local_search = None
        self._global_search = None
        self._hybrid_search = None
        self._answer_generator = None
        
        self._loaded = False
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return self._default_config()
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'embedding': {'model_name': 'sentence-transformers/all-MiniLM-L6-v2', 'device': 'cpu'},
            'llm': {'model_name': 'granite3-dense:2b', 'base_url': 'http://localhost:11434'},
            'retrieval': {
                'local': {'entity_threshold': 0.6, 'document_threshold': 0.4, 'top_k': 5},
                'global': {'community_top_k': 3, 'chunk_top_k': 5},
                'hybrid': {'local_weight': 0.6, 'global_weight': 0.4}
            },
            'paths': {
                'chunks_file': 'data/processed/chunks.json',
                'entities_file': 'data/processed/entities.json',
                'graph_file': 'data/processed/knowledge_graph.pkl',
                'embeddings_file': 'data/processed/embeddings.npz',
                'community_summaries_file': 'data/processed/community_summaries.json',
                'communities_file': 'data/processed/communities.json'
            }
        }
    
    def _setup_logging(self):
        """Setup logging."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        logging.basicConfig(level=level)
    
    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.config['embedding']['model_name']
            device = self.config['embedding'].get('device', 'cpu')
            logger.info(f"Loading embedding model: {model_name}")
            self._embedding_model = SentenceTransformer(model_name, device=device)
        return self._embedding_model
    
    @property
    def graph_builder(self):
        """Lazy load graph builder."""
        if self._graph_builder is None:
            from ..graph.graph_builder import KnowledgeGraphBuilder
            self._graph_builder = KnowledgeGraphBuilder()
        return self._graph_builder
    
    @property
    def local_search(self):
        """Lazy load local search."""
        if self._local_search is None:
            from ..retrieval.local_search import LocalGraphSearch
            config = self.config.get('retrieval', {}).get('local', {})
            self._local_search = LocalGraphSearch(
                entity_threshold=config.get('entity_threshold', 0.6),
                document_threshold=config.get('document_threshold', 0.4),
                top_k=config.get('top_k', 5)
            )
            self._local_search.set_embedding_model(self.embedding_model)
        return self._local_search
    
    @property
    def global_search(self):
        """Lazy load global search."""
        if self._global_search is None:
            from ..retrieval.global_search import GlobalGraphSearch
            config = self.config.get('retrieval', {}).get('global', {})
            self._global_search = GlobalGraphSearch(
                community_top_k=config.get('community_top_k', 3),
                chunk_top_k=config.get('chunk_top_k', 5)
            )
            self._global_search.set_embedding_model(self.embedding_model)
        return self._global_search
    
    @property
    def hybrid_search(self):
        """Lazy load hybrid search."""
        if self._hybrid_search is None:
            from ..retrieval.hybrid_search import HybridSearch
            config = self.config.get('retrieval', {}).get('hybrid', {})
            self._hybrid_search = HybridSearch(
                local_search=self.local_search,
                global_search=self.global_search,
                local_weight=config.get('local_weight', 0.6),
                global_weight=config.get('global_weight', 0.4)
            )
        return self._hybrid_search
    
    @property
    def answer_generator(self):
        """Lazy load answer generator."""
        if self._answer_generator is None:
            from ..llm.answer_generator import create_answer_generator
            llm_config = self.config.get('llm', {})
            self._answer_generator = create_answer_generator(
                model_name=llm_config.get('model_name', 'granite3-dense:2b'),
                base_url=llm_config.get('base_url', 'http://localhost:11434')
            )
        return self._answer_generator
    
    def load_indices(self) -> bool:
        """
        Load all pre-built indices.
        
        Returns:
            True if successful
        """
        logger.info("Loading indices...")
        paths = self.config['paths']
        
        try:
            # Load chunks
            chunks_path = Path(paths['chunks_file'])
            if chunks_path.exists():
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                logger.info(f"Loaded {len(self.chunks)} chunks")
            else:
                logger.error(f"Chunks file not found: {chunks_path}")
                return False
            
            # Load entities
            entities_path = Path(paths['entities_file'])
            if entities_path.exists():
                with open(entities_path, 'r', encoding='utf-8') as f:
                    self.entities = json.load(f)
                logger.info(f"Loaded {len(self.entities)} entities")
                
                # Build entity to chunks mapping
                for entity in self.entities:
                    normalized = entity.get('normalized', entity.get('text', ''))
                    self.entity_to_chunks[normalized] = entity.get('chunk_ids', [])
            
            # Load embeddings - try multiple possible paths
            embeddings_path = Path(paths['embeddings_file'])
            possible_paths = [
                embeddings_path,
                Path('data/processed/embeddings.npz'),
                Path('data/processed/embeddings.npy.npz'),
                Path(str(embeddings_path).replace('.npz', '.npy.npz')),
            ]
            
            embeddings_loaded = False
            for try_path in possible_paths:
                if try_path.exists():
                    try:
                        data = np.load(try_path)
                        self.chunk_embeddings = data['chunk_embeddings']
                        self.entity_embeddings = data['entity_embeddings']
                        logger.info(f"Loaded embeddings from {try_path}: "
                                   f"{self.chunk_embeddings.shape[0]} chunks, "
                                   f"{self.entity_embeddings.shape[0]} entities")
                        embeddings_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {try_path}: {e}")
                        continue
            
            if not embeddings_loaded:
                logger.error(f"Embeddings file not found. Tried: {[str(p) for p in possible_paths]}")
                return False
            
            # Load knowledge graph
            graph_path = Path(paths['graph_file'])
            if graph_path.exists():
                self.graph_builder.load(graph_path)
                logger.info("Loaded knowledge graph")
            
            # Load community summaries
            summaries_path = Path(paths['community_summaries_file'])
            if summaries_path.exists():
                with open(summaries_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.community_summaries = {int(k): v for k, v in data.items()}
                logger.info(f"Loaded {len(self.community_summaries)} community summaries")
            
            # Load communities info
            communities_path = Path(paths.get('communities_file', 'data/processed/communities.json'))
            if communities_path.exists():
                with open(communities_path, 'r', encoding='utf-8') as f:
                    self.communities_info = json.load(f)
                
                # Build community to chunks mapping
                for comm in self.communities_info:
                    comm_id = comm.get('id')
                    chunk_ids = set()
                    for entity in comm.get('entities', []):
                        chunk_ids.update(entity.get('chunk_ids', []))
                    self.community_to_chunks[comm_id] = list(chunk_ids)
            
            # Index for search
            self._index_for_search()
            
            self._loaded = True
            logger.info("All indices loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            return False
    
    def _index_for_search(self):
        """Index loaded data for search components."""
        # Index for local search
        self.local_search.index(
            entities=self.entities,
            chunks=self.chunks,
            entity_embeddings=self.entity_embeddings,
            chunk_embeddings=self.chunk_embeddings,
            entity_to_chunks=self.entity_to_chunks
        )
        
        # Index for global search
        self.global_search.index(
            communities=self.communities_info,
            summaries=self.community_summaries,
            community_to_chunks=self.community_to_chunks,
            chunks=self.chunks,
            chunk_embeddings=self.chunk_embeddings
        )
    
    def query(
        self,
        question: str,
        search_type: str = "hybrid",
        top_k: int = 5,
        include_sources: bool = True,
        stream: bool = False
    ):
        """
        Query the system and get an answer.
        
        Args:
            question: User question
            search_type: "local", "global", or "hybrid"
            top_k: Number of results
            include_sources: Include source citations
            stream: Stream the response
            
        Returns:
            Answer dict or generator if streaming
        """
        if not self._loaded:
            if not self.load_indices():
                return {"error": "Failed to load indices", "success": False}
        
        logger.info(f"Query: '{question[:50]}...' (type: {search_type})")
        
        # Retrieve relevant context
        if search_type == "local":
            results = self.local_search.search(question, top_k=top_k)
        elif search_type == "global":
            results = self.global_search.search(question, chunk_top_k=top_k)
        else:  # hybrid
            results = self.hybrid_search.search(question, top_k=top_k)
        
        logger.info(f"Retrieved {len(results)} results")
        
        # Generate answer
        response = self.answer_generator.generate(
            query=question,
            retrieved_results=results,
            stream=stream
        )
        
        return response
    
    def search_only(
        self,
        question: str,
        search_type: str = "hybrid",
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search without generating an answer.
        
        Args:
            question: Search query
            search_type: "local", "global", or "hybrid"
            top_k: Number of results
            
        Returns:
            List of search results
        """
        if not self._loaded:
            if not self.load_indices():
                return []
        
        if search_type == "local":
            return self.local_search.search(question, top_k=top_k)
        elif search_type == "global":
            return self.global_search.search(question, chunk_top_k=top_k)
        else:
            return self.hybrid_search.search(question, top_k=top_k)
    
    def get_entity_info(self, entity_text: str) -> Dict:
        """Get information about an entity."""
        for entity in self.entities:
            if entity.get('normalized', '').lower() == entity_text.lower():
                return entity
        return {}
    
    def get_community_info(self, community_id: int) -> Dict:
        """Get information about a community."""
        for comm in self.communities_info:
            if comm.get('id') == community_id:
                return {
                    **comm,
                    'summary': self.community_summaries.get(community_id, '')
                }
        return {}
    
    def clear_history(self):
        """Clear conversation history."""
        if self._answer_generator:
            self._answer_generator.clear_history()
    
    def get_statistics(self) -> Dict:
        """Get system statistics."""
        return {
            'loaded': self._loaded,
            'num_chunks': len(self.chunks),
            'num_entities': len(self.entities),
            'num_communities': len(self.communities_info),
            'graph_nodes': self.graph_builder.graph.number_of_nodes() if self._graph_builder else 0,
            'graph_edges': self.graph_builder.graph.number_of_edges() if self._graph_builder else 0
        }


def create_query_engine(config_path: str = "config.yaml") -> QueryEngine:
    """
    Factory function to create and load query engine.
    
    Args:
        config_path: Configuration file path
        
    Returns:
        Loaded QueryEngine instance
    """
    engine = QueryEngine(config_path=config_path)
    engine.load_indices()
    return engine
