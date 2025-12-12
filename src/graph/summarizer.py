"""
Community Summarizer - Generate LLM summaries for communities

Creates natural language summaries of entity communities
using the local LLM, which are used for global search.
"""

import logging
from typing import List, Dict, Optional
import json

import networkx as nx

logger = logging.getLogger(__name__)


class CommunitySummarizer:
    """
    Generates summaries for entity communities using LLM.
    
    These summaries capture the theme and key information
    of each community for global RAG search.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the summarizer.
        
        Args:
            llm_client: LLM client for generating summaries
        """
        self.llm_client = llm_client
        self.summaries = {}  # community_id -> summary
        
    def set_llm_client(self, llm_client):
        """Set the LLM client after initialization."""
        self.llm_client = llm_client
    
    def generate_summary(
        self,
        community_info: Dict,
        chunks: List[Dict],
        graph: nx.Graph
    ) -> str:
        """
        Generate a summary for a single community.
        
        Args:
            community_info: Dict with community information
            chunks: List of relevant chunks
            graph: Knowledge graph
            
        Returns:
            Generated summary string
        """
        if not self.llm_client:
            # Fallback to rule-based summary
            return self._generate_fallback_summary(community_info)
        
        # Build context from community info
        entities = community_info.get('entities', [])
        entity_texts = [e['text'] for e in entities[:10]]
        
        # Get relevant chunk texts
        chunk_texts = [c.get('text', '')[:500] for c in chunks[:3]]
        
        prompt = self._build_summary_prompt(entity_texts, chunk_texts, graph, community_info)
        
        try:
            summary = self.llm_client.generate(prompt)
            return summary.strip()
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}, using fallback")
            return self._generate_fallback_summary(community_info)
    
    def _build_summary_prompt(
        self,
        entities: List[str],
        chunk_texts: List[str],
        graph: nx.Graph,
        community_info: Dict
    ) -> str:
        """Build prompt for community summarization."""
        
        # Get relationships within community
        relationships = []
        nodes = community_info.get('entities', [])[:10]
        node_ids = [n.get('node_id') for n in nodes if n.get('node_id')]
        
        for i, nid1 in enumerate(node_ids):
            for nid2 in node_ids[i+1:]:
                if graph.has_edge(nid1, nid2):
                    edge_data = graph[nid1][nid2]
                    rel_types = edge_data.get('relation_types', ['RELATED'])
                    n1_text = graph.nodes[nid1].get('text', nid1)
                    n2_text = graph.nodes[nid2].get('text', nid2)
                    relationships.append(f"- {n1_text} {rel_types[0]} {n2_text}")
        
        prompt = f"""Analyze the following group of related entities and text excerpts to create a comprehensive summary.

ENTITIES IN THIS GROUP:
{', '.join(entities)}

RELATIONSHIPS:
{chr(10).join(relationships[:5]) if relationships else 'No direct relationships found'}

RELEVANT TEXT EXCERPTS:
{chr(10).join(f'- {text[:300]}...' for text in chunk_texts if text)}

Create a 2-3 sentence summary that:
1. Identifies the main theme or topic of this group
2. Highlights key entities and their relationships
3. Captures the most important information

Summary:"""
        
        return prompt
    
    def _generate_fallback_summary(self, community_info: Dict) -> str:
        """Generate a rule-based summary without LLM."""
        entities = community_info.get('entities', [])
        top_entities = community_info.get('top_entities', [])
        
        if not entities:
            return "Empty community with no entities."
        
        # Group by entity type
        by_type = {}
        for e in entities:
            label = e.get('label', 'UNKNOWN')
            if label not in by_type:
                by_type[label] = []
            by_type[label].append(e['text'])
        
        # Build summary
        parts = []
        
        if 'PERSON' in by_type:
            persons = by_type['PERSON'][:3]
            parts.append(f"Key figures: {', '.join(persons)}")
        
        if 'ORG' in by_type:
            orgs = by_type['ORG'][:3]
            parts.append(f"Organizations: {', '.join(orgs)}")
        
        if 'GPE' in by_type or 'LOC' in by_type:
            places = by_type.get('GPE', []) + by_type.get('LOC', [])
            parts.append(f"Places: {', '.join(places[:3])}")
        
        if 'EVENT' in by_type:
            events = by_type['EVENT'][:3]
            parts.append(f"Events: {', '.join(events)}")
        
        if not parts:
            parts.append(f"Entities: {', '.join(top_entities[:5])}")
        
        summary = f"This community contains {len(entities)} related entities. {'. '.join(parts)}."
        
        return summary
    
    def summarize_all_communities(
        self,
        communities_info: List[Dict],
        chunks: List[Dict],
        graph: nx.Graph,
        chunk_to_community: Optional[Dict] = None
    ) -> Dict[int, str]:
        """
        Generate summaries for all communities.
        
        Args:
            communities_info: List of community info dicts
            chunks: All chunks
            graph: Knowledge graph
            chunk_to_community: Optional mapping from chunk ID to community
            
        Returns:
            Dict mapping community ID to summary
        """
        logger.info(f"Generating summaries for {len(communities_info)} communities...")
        
        self.summaries = {}
        
        for i, comm_info in enumerate(communities_info):
            comm_id = comm_info.get('id', i)
            
            # Find relevant chunks for this community
            relevant_chunks = self._get_community_chunks(
                comm_info, chunks, chunk_to_community
            )
            
            summary = self.generate_summary(comm_info, relevant_chunks, graph)
            self.summaries[comm_id] = summary
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{len(communities_info)} summaries")
        
        logger.info(f"Generated {len(self.summaries)} community summaries")
        
        return self.summaries
    
    def _get_community_chunks(
        self,
        community_info: Dict,
        all_chunks: List[Dict],
        chunk_to_community: Optional[Dict]
    ) -> List[Dict]:
        """Get chunks relevant to a community."""
        
        # Get chunk IDs from entities
        chunk_ids = set()
        for entity in community_info.get('entities', []):
            for cid in entity.get('chunk_ids', []):
                chunk_ids.add(cid)
        
        # Filter chunks
        if chunk_ids:
            relevant = [c for c in all_chunks if c.get('id') in chunk_ids]
        else:
            # Fallback: use first few chunks
            relevant = all_chunks[:3]
        
        return relevant[:5]  # Limit for context
    
    def get_summary(self, community_id: int) -> Optional[str]:
        """Get summary for a specific community."""
        return self.summaries.get(community_id)
    
    def get_all_summaries(self) -> Dict[int, str]:
        """Get all community summaries."""
        return self.summaries.copy()
    
    def save_summaries(self, path: str):
        """Save summaries to JSON file."""
        import json
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert keys to strings for JSON
        data = {str(k): v for k, v in self.summaries.items()}
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.summaries)} summaries to {path}")
    
    def load_summaries(self, path: str):
        """Load summaries from JSON file."""
        from pathlib import Path
        
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Summaries file not found: {path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert keys back to integers
        self.summaries = {int(k): v for k, v in data.items()}
        
        logger.info(f"Loaded {len(self.summaries)} summaries from {path}")
