"""
Local Graph RAG Search - Equation 4 from SemRAG Paper

Implements local search by:
1. Finding entities similar to query
2. Retrieving chunks containing those entities
3. Ranking by combined similarity scores
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class LocalGraphSearch:
    """
    Implements Local Graph RAG Search (Equation 4 from SemRAG).
    
    D_retrieved = Top_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d})
    
    Where:
    - V: Entity set
    - G: Chunk set
    - Q: Query
    - H: History (optional)
    - τ_e: Entity similarity threshold
    - τ_d: Document similarity threshold
    """
    
    def __init__(
        self,
        entity_threshold: float = 0.6,
        document_threshold: float = 0.4,
        top_k: int = 5
    ):
        """
        Initialize local search.
        
        Args:
            entity_threshold: τ_e - threshold for entity similarity
            document_threshold: τ_d - threshold for chunk similarity
            top_k: Number of results to return
        """
        self.entity_threshold = entity_threshold
        self.document_threshold = document_threshold
        self.top_k = top_k
        
        # Will be set during indexing
        self.entity_embeddings = None  # np.array of entity embeddings
        self.entity_ids = []  # List of entity IDs
        self.entity_info = {}  # Dict mapping entity ID to info
        self.chunk_embeddings = None  # np.array of chunk embeddings
        self.chunk_ids = []  # List of chunk IDs
        self.chunk_info = {}  # Dict mapping chunk ID to info
        self.entity_to_chunks = {}  # Mapping from entity to chunk IDs
        
        self.embedding_model = None
    
    def set_embedding_model(self, model):
        """Set the embedding model for query encoding."""
        self.embedding_model = model
    
    def index(
        self,
        entities: List[Dict],
        chunks: List[Dict],
        entity_embeddings: np.ndarray,
        chunk_embeddings: np.ndarray,
        entity_to_chunks: Dict[str, List[str]]
    ):
        """
        Index entities and chunks for search.
        
        Args:
            entities: List of entity dicts
            chunks: List of chunk dicts
            entity_embeddings: Entity embedding matrix
            chunk_embeddings: Chunk embedding matrix
            entity_to_chunks: Mapping from entity to chunk IDs
        """
        logger.info("Indexing for local search...")
        
        self.entity_ids = [e.get('normalized', e.get('id', str(i))) 
                         for i, e in enumerate(entities)]
        self.entity_info = {eid: e for eid, e in zip(self.entity_ids, entities)}
        self.entity_embeddings = entity_embeddings
        
        self.chunk_ids = [c.get('id', str(i)) for i, c in enumerate(chunks)]
        self.chunk_info = {cid: c for cid, c in zip(self.chunk_ids, chunks)}
        self.chunk_embeddings = chunk_embeddings
        
        self.entity_to_chunks = entity_to_chunks
        
        logger.info(f"Indexed {len(self.entity_ids)} entities and {len(self.chunk_ids)} chunks")
    
    def _encode_query(self, query: str, history: Optional[str] = None) -> np.ndarray:
        """
        Encode query (and optional history) into embedding.
        
        Args:
            query: Query string
            history: Optional conversation history
            
        Returns:
            Query embedding
        """
        if history:
            combined = f"{history} {query}"
        else:
            combined = query
        
        if self.embedding_model:
            embedding = self.embedding_model.encode([combined], convert_to_numpy=True)
            return embedding[0]
        else:
            raise ValueError("Embedding model not set. Call set_embedding_model() first.")
    
    def search(
        self,
        query: str,
        history: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Perform local graph RAG search.
        
        Implements Equation 4:
        D_retrieved = Top_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d})
        
        Args:
            query: Search query
            history: Optional conversation history
            top_k: Override default top_k
            
        Returns:
            List of result dicts with chunk info and scores
        """
        if self.entity_embeddings is None or self.chunk_embeddings is None:
            raise ValueError("Index not built. Call index() first.")
        
        top_k = top_k or self.top_k
        
        logger.debug(f"Local search: '{query[:50]}...'")
        
        # Step 1: Encode query
        query_embedding = self._encode_query(query, history)
        
        # Step 2: Calculate similarity between query and all entities
        entity_similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.entity_embeddings
        )[0]
        
        # Step 3: Filter entities by threshold τ_e
        relevant_entities = []
        for i, sim in enumerate(entity_similarities):
            if sim >= self.entity_threshold:
                relevant_entities.append({
                    'id': self.entity_ids[i],
                    'similarity': float(sim),
                    'info': self.entity_info.get(self.entity_ids[i], {})
                })
        
        logger.debug(f"Found {len(relevant_entities)} entities above threshold")
        
        if not relevant_entities:
            # Lower threshold if no entities found
            threshold = self.entity_threshold * 0.7
            for i, sim in enumerate(entity_similarities):
                if sim >= threshold:
                    relevant_entities.append({
                        'id': self.entity_ids[i],
                        'similarity': float(sim),
                        'info': self.entity_info.get(self.entity_ids[i], {})
                    })
            logger.debug(f"Lowered threshold, found {len(relevant_entities)} entities")
        
        # Step 4: Find chunks related to filtered entities
        candidate_chunks = {}
        for entity in relevant_entities:
            entity_id = entity['id']
            chunk_ids = self.entity_to_chunks.get(entity_id, [])
            
            for chunk_id in chunk_ids:
                if chunk_id not in candidate_chunks:
                    candidate_chunks[chunk_id] = {
                        'entities': [],
                        'entity_scores': []
                    }
                candidate_chunks[chunk_id]['entities'].append(entity_id)
                candidate_chunks[chunk_id]['entity_scores'].append(entity['similarity'])
        
        logger.debug(f"Found {len(candidate_chunks)} candidate chunks")
        
        # Step 5: Calculate chunk similarities and filter by τ_d
        results = []
        
        for chunk_id, chunk_data in candidate_chunks.items():
            # Get chunk index
            try:
                chunk_idx = self.chunk_ids.index(chunk_id)
            except ValueError:
                continue
            
            # Calculate chunk-query similarity
            chunk_sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.chunk_embeddings[chunk_idx].reshape(1, -1)
            )[0][0]
            
            if chunk_sim >= self.document_threshold:
                # Combine scores: entity relevance + chunk similarity
                entity_score = np.mean(chunk_data['entity_scores'])
                combined_score = 0.6 * chunk_sim + 0.4 * entity_score
                
                chunk_info = self.chunk_info.get(chunk_id, {})
                
                results.append({
                    'chunk_id': chunk_id,
                    'text': chunk_info.get('text', ''),
                    'chunk_similarity': float(chunk_sim),
                    'entity_similarity': float(entity_score),
                    'combined_score': float(combined_score),
                    'entities': chunk_data['entities'],
                    'metadata': chunk_info.get('metadata', {})
                })
        
        # Step 6: Sort by combined score and return top_k
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        results = results[:top_k]
        
        logger.debug(f"Returning {len(results)} results")
        
        return results
    
    def search_with_entity_expansion(
        self,
        query: str,
        graph,
        history: Optional[str] = None,
        expansion_hops: int = 1,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Enhanced search with entity expansion via graph neighbors.
        
        Args:
            query: Search query
            graph: Knowledge graph builder
            history: Optional conversation history
            expansion_hops: Number of hops for entity expansion
            top_k: Override default top_k
            
        Returns:
            List of result dicts
        """
        # First do regular search
        results = self.search(query, history, top_k)
        
        if not results or not graph:
            return results
        
        # Get entities from results
        found_entities = set()
        for result in results:
            found_entities.update(result.get('entities', []))
        
        # Expand with neighbors
        expanded_chunks = set()
        for entity in found_entities:
            neighbors = graph.get_neighbors(entity, max_depth=expansion_hops)
            for neighbor in neighbors:
                neighbor_chunks = graph.get_chunks_for_entity(neighbor['text'])
                expanded_chunks.update(neighbor_chunks)
        
        # Add expanded chunks if not already in results
        existing_chunks = {r['chunk_id'] for r in results}
        
        for chunk_id in expanded_chunks:
            if chunk_id in existing_chunks:
                continue
            
            chunk_info = self.chunk_info.get(chunk_id)
            if chunk_info:
                results.append({
                    'chunk_id': chunk_id,
                    'text': chunk_info.get('text', ''),
                    'combined_score': 0.3,  # Lower score for expanded
                    'entities': [],
                    'metadata': {'expanded': True}
                })
        
        return results[:top_k or self.top_k]
