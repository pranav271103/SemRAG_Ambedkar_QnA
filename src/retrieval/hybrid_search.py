"""
Hybrid Search - Combines Local and Global RAG Search

Merges results from both search strategies for comprehensive retrieval.
"""

import logging
from typing import List, Dict, Optional

from .local_search import LocalGraphSearch
from .global_search import GlobalGraphSearch

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Combines Local and Global Graph RAG Search.
    
    Local search finds specific entities and their related chunks.
    Global search finds thematically related communities and chunks.
    
    The hybrid approach merges both for comprehensive coverage.
    """
    
    def __init__(
        self,
        local_search: LocalGraphSearch,
        global_search: GlobalGraphSearch,
        local_weight: float = 0.6,
        global_weight: float = 0.4
    ):
        """
        Initialize hybrid search.
        
        Args:
            local_search: Local search instance
            global_search: Global search instance
            local_weight: Weight for local search results
            global_weight: Weight for global search results
        """
        self.local_search = local_search
        self.global_search = global_search
        self.local_weight = local_weight
        self.global_weight = global_weight
    
    def search(
        self,
        query: str,
        history: Optional[str] = None,
        top_k: int = 5,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Perform hybrid search combining local and global results.
        
        Args:
            query: Search query
            history: Optional conversation history
            top_k: Number of results to return
            include_metadata: Include search metadata
            
        Returns:
            List of merged and ranked results
        """
        logger.info(f"Hybrid search: '{query[:50]}...'")
        
        # Get local search results
        try:
            local_results = self.local_search.search(
                query=query,
                history=history,
                top_k=top_k * 2  # Get more to allow merging
            )
        except Exception as e:
            logger.warning(f"Local search failed: {e}")
            local_results = []
        
        # Get global search results
        try:
            global_results = self.global_search.search(
                query=query,
                chunk_top_k=top_k * 2
            )
        except Exception as e:
            logger.warning(f"Global search failed: {e}")
            global_results = []
        
        # Merge results
        merged = self._merge_results(
            local_results,
            global_results,
            include_metadata
        )
        
        # Sort by combined score
        merged.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Return top_k
        results = merged[:top_k]
        
        logger.info(f"Hybrid search returned {len(results)} results "
                   f"(local: {len(local_results)}, global: {len(global_results)})")
        
        return results
    
    def _merge_results(
        self,
        local_results: List[Dict],
        global_results: List[Dict],
        include_metadata: bool
    ) -> List[Dict]:
        """
        Merge local and global search results.
        
        Handles deduplication and score combination.
        
        Args:
            local_results: Results from local search
            global_results: Results from global search
            include_metadata: Include source metadata
            
        Returns:
            Merged results list
        """
        merged = {}
        
        # Process local results
        for result in local_results:
            chunk_id = result.get('chunk_id')
            if not chunk_id:
                continue
            
            local_score = result.get('combined_score', 0)
            
            merged[chunk_id] = {
                'chunk_id': chunk_id,
                'text': result.get('text', ''),
                'local_score': local_score,
                'global_score': 0,
                'entities': result.get('entities', []),
                'communities': [],
                'community_summary': '',
                'metadata': result.get('metadata', {})
            }
        
        # Process global results
        for result in global_results:
            chunk_id = result.get('chunk_id')
            if not chunk_id:
                continue
            
            global_score = result.get('combined_score', 0)
            
            if chunk_id in merged:
                # Update existing entry
                merged[chunk_id]['global_score'] = global_score
                merged[chunk_id]['communities'] = result.get('communities', [])
                merged[chunk_id]['community_summary'] = result.get('community_summary', '')
            else:
                # Add new entry
                merged[chunk_id] = {
                    'chunk_id': chunk_id,
                    'text': result.get('text', ''),
                    'local_score': 0,
                    'global_score': global_score,
                    'entities': [],
                    'communities': result.get('communities', []),
                    'community_summary': result.get('community_summary', ''),
                    'metadata': result.get('metadata', {})
                }
        
        # Calculate final scores
        results = []
        for chunk_id, data in merged.items():
            # Weighted combination
            final_score = (
                self.local_weight * data['local_score'] +
                self.global_weight * data['global_score']
            )
            
            result = {
                'chunk_id': chunk_id,
                'text': data['text'],
                'final_score': final_score,
                'entities': data['entities'],
                'communities': data['communities'],
                'community_summary': data['community_summary']
            }
            
            if include_metadata:
                result['metadata'] = {
                    'local_score': data['local_score'],
                    'global_score': data['global_score'],
                    'source': self._determine_source(data)
                }
            
            results.append(result)
        
        return results
    
    def _determine_source(self, data: Dict) -> str:
        """Determine which search method contributed more."""
        if data['local_score'] > 0 and data['global_score'] > 0:
            return 'both'
        elif data['local_score'] > 0:
            return 'local'
        else:
            return 'global'
    
    def search_with_explanation(
        self,
        query: str,
        history: Optional[str] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Search with detailed explanation of results.
        
        Args:
            query: Search query
            history: Optional conversation history
            top_k: Number of results
            
        Returns:
            Dict with results and explanation
        """
        results = self.search(query, history, top_k, include_metadata=True)
        
        # Analyze result sources
        source_counts = {'local': 0, 'global': 0, 'both': 0}
        for r in results:
            source = r.get('metadata', {}).get('source', 'unknown')
            if source in source_counts:
                source_counts[source] += 1
        
        # Get community overview
        community_overview = []
        try:
            community_overview = self.global_search.get_community_overview(query, top_k=3)
        except Exception:
            pass
        
        # Collect unique entities
        all_entities = set()
        for r in results:
            all_entities.update(r.get('entities', []))
        
        explanation = {
            'query': query,
            'results': results,
            'analysis': {
                'total_results': len(results),
                'source_distribution': source_counts,
                'relevant_entities': list(all_entities)[:10],
                'relevant_communities': community_overview
            }
        }
        
        return explanation
