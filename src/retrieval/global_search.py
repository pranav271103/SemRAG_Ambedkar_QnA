"""
Global Graph RAG Search - Equation 5 from SemRAG Paper

Implements global search by:
1. Finding communities relevant to query
2. Using community summaries for retrieval
3. Aggregating chunks from matched communities
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class GlobalGraphSearch:
    """
    Implements Global Graph RAG Search (Equation 5 from SemRAG).
    
    D_retrieved = Top_k(⋃_{r ∈ R_Top-K(Q)} ⋃_{c_i ∈ C_r} (⋃_{p_j ∈ c_i} (p_j, score(p_j, Q))))
    
    Where:
    - R: Community reports/summaries
    - C_r: Chunks in community r
    - p_j: Points (sentences/paragraphs) within chunks
    - Q: Query
    """
    
    def __init__(
        self,
        community_top_k: int = 3,
        chunk_top_k: int = 5
    ):
        """
        Initialize global search.
        
        Args:
            community_top_k: Number of top communities to consider
            chunk_top_k: Number of top chunks to return
        """
        self.community_top_k = community_top_k
        self.chunk_top_k = chunk_top_k
        
        # Will be set during indexing
        self.community_embeddings = None  # np.array of community summary embeddings
        self.community_ids = []  # List of community IDs
        self.community_summaries = {}  # community_id -> summary text
        self.community_chunks = {}  # community_id -> list of chunk IDs
        self.chunk_embeddings = None
        self.chunk_ids = []
        self.chunk_info = {}
        
        self.embedding_model = None
    
    def set_embedding_model(self, model):
        """Set the embedding model for query encoding."""
        self.embedding_model = model
    
    def index(
        self,
        communities: List[Dict],
        summaries: Dict[int, str],
        community_to_chunks: Dict[int, List[str]],
        chunks: List[Dict],
        chunk_embeddings: np.ndarray
    ):
        """
        Index communities and chunks for global search.
        
        Args:
            communities: List of community info dicts
            summaries: Mapping from community ID to summary
            community_to_chunks: Mapping from community ID to chunk IDs
            chunks: List of chunk dicts
            chunk_embeddings: Chunk embedding matrix
        """
        logger.info("Indexing for global search...")
        
        # Store community info
        self.community_ids = [c.get('id', i) for i, c in enumerate(communities)]
        self.community_summaries = summaries
        self.community_chunks = community_to_chunks
        
        # Create community embeddings from summaries
        if self.embedding_model:
            summary_texts = [summaries.get(cid, '') for cid in self.community_ids]
            # Filter out empty summaries
            valid_indices = [i for i, s in enumerate(summary_texts) if s]
            valid_summaries = [summary_texts[i] for i in valid_indices]
            
            if valid_summaries:
                embeddings = self.embedding_model.encode(
                    valid_summaries,
                    convert_to_numpy=True
                )
                
                # Create full embedding matrix (with zeros for empty summaries)
                self.community_embeddings = np.zeros(
                    (len(summary_texts), embeddings.shape[1])
                )
                for i, idx in enumerate(valid_indices):
                    self.community_embeddings[idx] = embeddings[i]
            else:
                logger.warning("No valid community summaries to embed")
                self.community_embeddings = None
        
        # Store chunk info
        self.chunk_ids = [c.get('id', str(i)) for i, c in enumerate(chunks)]
        self.chunk_info = {cid: c for cid, c in zip(self.chunk_ids, chunks)}
        self.chunk_embeddings = chunk_embeddings
        
        logger.info(f"Indexed {len(self.community_ids)} communities and "
                   f"{len(self.chunk_ids)} chunks for global search")
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query into embedding."""
        if self.embedding_model:
            embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            return embedding[0]
        else:
            raise ValueError("Embedding model not set.")
    
    def search(
        self,
        query: str,
        community_top_k: Optional[int] = None,
        chunk_top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Perform global graph RAG search.
        
        Implements Equation 5:
        1. Find top-K communities relevant to query
        2. Get all chunks from those communities
        3. Score and rank chunks
        4. Return top chunks with community context
        
        Args:
            query: Search query
            community_top_k: Override number of communities
            chunk_top_k: Override number of chunks to return
            
        Returns:
            List of result dicts with chunk info, scores, and community context
        """
        community_top_k = community_top_k or self.community_top_k
        chunk_top_k = chunk_top_k or self.chunk_top_k
        
        logger.debug(f"Global search: '{query[:50]}...'")
        
        if self.community_embeddings is None:
            logger.warning("Community embeddings not indexed")
            return []
        
        # Step 1: Encode query
        query_embedding = self._encode_query(query)
        
        # Step 2: Find top-K communities by summary similarity
        community_similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.community_embeddings
        )[0]
        
        # Get top communities
        top_indices = np.argsort(community_similarities)[::-1][:community_top_k]
        top_communities = []
        
        for idx in top_indices:
            comm_id = self.community_ids[idx]
            sim = community_similarities[idx]
            
            if sim > 0.1:  # Minimum threshold
                top_communities.append({
                    'id': comm_id,
                    'similarity': float(sim),
                    'summary': self.community_summaries.get(comm_id, '')
                })
        
        logger.debug(f"Found {len(top_communities)} relevant communities")
        
        if not top_communities:
            return []
        
        # Step 3: Get all chunks from top communities
        candidate_chunks = {}
        
        for comm in top_communities:
            comm_id = comm['id']
            chunk_ids = self.community_chunks.get(comm_id, [])
            
            for chunk_id in chunk_ids:
                if chunk_id not in candidate_chunks:
                    candidate_chunks[chunk_id] = {
                        'communities': [],
                        'community_scores': []
                    }
                candidate_chunks[chunk_id]['communities'].append(comm_id)
                candidate_chunks[chunk_id]['community_scores'].append(comm['similarity'])
        
        logger.debug(f"Found {len(candidate_chunks)} candidate chunks from communities")
        
        # Step 4: Score each chunk against query
        results = []
        
        for chunk_id, chunk_data in candidate_chunks.items():
            try:
                chunk_idx = self.chunk_ids.index(chunk_id)
            except ValueError:
                continue
            
            # Calculate chunk-query similarity
            chunk_sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.chunk_embeddings[chunk_idx].reshape(1, -1)
            )[0][0]
            
            # Combined score: chunk similarity + community relevance
            community_score = np.max(chunk_data['community_scores'])
            combined_score = 0.5 * chunk_sim + 0.5 * community_score
            
            chunk_info = self.chunk_info.get(chunk_id, {})
            
            # Get community summary for context
            main_community = chunk_data['communities'][0]
            community_summary = self.community_summaries.get(main_community, '')
            
            results.append({
                'chunk_id': chunk_id,
                'text': chunk_info.get('text', ''),
                'chunk_similarity': float(chunk_sim),
                'community_similarity': float(community_score),
                'combined_score': float(combined_score),
                'communities': chunk_data['communities'],
                'community_summary': community_summary,
                'metadata': chunk_info.get('metadata', {})
            })
        
        # Step 5: Sort and return top chunks
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        results = results[:chunk_top_k]
        
        logger.debug(f"Returning {len(results)} global search results")
        
        return results
    
    def get_community_overview(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Get overview of communities relevant to query.
        
        Useful for understanding the high-level themes related to a query.
        
        Args:
            query: Search query
            top_k: Number of communities to return
            
        Returns:
            List of community info dicts
        """
        if self.community_embeddings is None:
            return []
        
        query_embedding = self._encode_query(query)
        
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.community_embeddings
        )[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            comm_id = self.community_ids[idx]
            results.append({
                'community_id': comm_id,
                'similarity': float(similarities[idx]),
                'summary': self.community_summaries.get(comm_id, ''),
                'num_chunks': len(self.community_chunks.get(comm_id, []))
            })
        
        return results
