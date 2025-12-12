"""
Indexer Pipeline - Processes PDF and builds all indices

Main pipeline that:
1. Extracts text from PDF
2. Creates semantic chunks
3. Builds knowledge graph
4. Detects communities
5. Generates community summaries
6. Saves all indices for retrieval
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional
import yaml
import numpy as np

logger = logging.getLogger(__name__)


class Indexer:
    """
    Main indexing pipeline for SemRAG.
    
    Processes the input PDF and creates all necessary indices
    for the retrieval system.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize indexer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Components (lazy loaded)
        self._embedding_model = None
        self._chunker = None
        self._entity_extractor = None
        self._relationship_extractor = None
        self._graph_builder = None
        self._community_detector = None
        self._summarizer = None
        self._llm_client = None
        
        # Data stores
        self.chunks = []
        self.entities = []
        self.relationships = []
        self.communities_info = []
        self.chunk_embeddings = None
        self.entity_embeddings = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._default_config()
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'embedding': {'model_name': 'sentence-transformers/all-MiniLM-L6-v2', 'device': 'cpu'},
            'llm': {'model_name': 'granite3-dense:2b', 'base_url': 'http://localhost:11434'},
            'chunking': {'similarity_threshold': 0.5, 'max_tokens': 1024, 'min_chunk_size': 50},
            'graph': {'min_entity_freq': 1, 'community_algorithm': 'louvain'},
            'paths': {
                'data_dir': 'data',
                'processed_dir': 'data/processed',
                'pdf_file': 'data/Ambedkar_book.pdf',
                'chunks_file': 'data/processed/chunks.json',
                'entities_file': 'data/processed/entities.json',
                'graph_file': 'data/processed/knowledge_graph.pkl',
                'embeddings_file': 'data/processed/embeddings.npz',
                'community_summaries_file': 'data/processed/community_summaries.json'
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(level=level, format=format_str)
    
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
    def chunker(self):
        """Lazy load chunker."""
        if self._chunker is None:
            from ..chunking.semantic_chunker import SemanticChunker
            chunk_config = self.config.get('chunking', {})
            self._chunker = SemanticChunker(
                model_name=self.config['embedding']['model_name'],
                similarity_threshold=chunk_config.get('similarity_threshold', 0.5),
                max_tokens=chunk_config.get('max_tokens', 1024),
                min_chunk_size=chunk_config.get('min_chunk_size', 50),
                device=self.config['embedding'].get('device', 'cpu')
            )
        return self._chunker
    
    @property
    def entity_extractor(self):
        """Lazy load entity extractor."""
        if self._entity_extractor is None:
            from ..graph.entity_extractor import EntityExtractor
            self._entity_extractor = EntityExtractor()
        return self._entity_extractor
    
    @property
    def relationship_extractor(self):
        """Lazy load relationship extractor."""
        if self._relationship_extractor is None:
            from ..graph.relationship_extractor import RelationshipExtractor
            self._relationship_extractor = RelationshipExtractor()
        return self._relationship_extractor
    
    @property
    def graph_builder(self):
        """Lazy load graph builder."""
        if self._graph_builder is None:
            from ..graph.graph_builder import KnowledgeGraphBuilder
            self._graph_builder = KnowledgeGraphBuilder()
        return self._graph_builder
    
    @property
    def community_detector(self):
        """Lazy load community detector."""
        if self._community_detector is None:
            from ..graph.community_detector import CommunityDetector
            graph_config = self.config.get('graph', {})
            self._community_detector = CommunityDetector(
                algorithm=graph_config.get('community_algorithm', 'louvain'),
                resolution=graph_config.get('community_resolution', 1.0)
            )
        return self._community_detector
    
    @property
    def summarizer(self):
        """Lazy load summarizer."""
        if self._summarizer is None:
            from ..graph.summarizer import CommunitySummarizer
            self._summarizer = CommunitySummarizer()
            if self._llm_client:
                self._summarizer.set_llm_client(self._llm_client)
        return self._summarizer
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            from ..llm.llm_client import OllamaClient
            llm_config = self.config.get('llm', {})
            self._llm_client = OllamaClient(
                model_name=llm_config.get('model_name', 'granite3-dense:2b'),
                base_url=llm_config.get('base_url', 'http://localhost:11434')
            )
            # Also set for summarizer
            if self._summarizer:
                self._summarizer.set_llm_client(self._llm_client)
        return self._llm_client
    
    def run(self, pdf_path: Optional[str] = None) -> Dict:
        """
        Run the full indexing pipeline.
        
        Args:
            pdf_path: Optional override for PDF path
            
        Returns:
            Dict with indexing statistics
        """
        pdf_path = pdf_path or self.config['paths']['pdf_file']
        
        logger.info("=" * 50)
        logger.info("Starting SemRAG Indexing Pipeline")
        logger.info("=" * 50)
        
        stats = {}
        
        # Step 1: Extract text from PDF
        logger.info("\n[Step 1/6] Extracting text from PDF...")
        from ..chunking.pdf_loader import extract_text_from_pdf
        text, pdf_metadata = extract_text_from_pdf(pdf_path)
        stats['pdf_pages'] = pdf_metadata.get('pages', 0)
        stats['text_length'] = len(text)
        logger.info(f"Extracted {len(text)} characters from {pdf_metadata.get('pages', '?')} pages")
        
        # Step 2: Create semantic chunks
        logger.info("\n[Step 2/6] Creating semantic chunks...")
        chunks = self.chunker.chunk(text, doc_id="ambedkar")
        self.chunks = [c.to_dict() for c in chunks]
        stats['num_chunks'] = len(self.chunks)
        logger.info(f"Created {len(self.chunks)} semantic chunks")
        
        # Step 3: Extract entities and relationships
        logger.info("\n[Step 3/6] Extracting entities and relationships...")
        entities, entity_to_chunks = self.entity_extractor.extract_from_chunks(self.chunks)
        self.entities = [e.to_dict() for e in entities]
        stats['num_entities'] = len(self.entities)
        logger.info(f"Extracted {len(self.entities)} unique entities")
        
        relationships = self.relationship_extractor.extract_from_chunks(
            self.chunks, self.entities
        )
        self.relationships = [r.to_dict() for r in relationships]
        stats['num_relationships'] = len(self.relationships)
        logger.info(f"Extracted {len(self.relationships)} relationships")
        
        # Step 4: Build knowledge graph
        logger.info("\n[Step 4/6] Building knowledge graph...")
        graph = self.graph_builder.build_graph(
            self.entities, self.relationships, self.chunks
        )
        graph_stats = self.graph_builder.get_statistics()
        stats['graph_nodes'] = graph_stats['num_nodes']
        stats['graph_edges'] = graph_stats['num_edges']
        
        # Step 5: Detect communities
        logger.info("\n[Step 5/6] Detecting communities...")
        self.community_detector.detect_communities(graph)
        self.communities_info = self.community_detector.get_all_communities_info(graph)
        community_stats = self.community_detector.get_statistics()
        stats['num_communities'] = community_stats['num_communities']
        logger.info(f"Detected {stats['num_communities']} communities")
        
        # Step 6: Generate community summaries
        logger.info("\n[Step 6/6] Generating community summaries...")
        # Build community to chunks mapping
        community_to_chunks = self._build_community_chunks_mapping()
        
        # Initialize LLM client for summarizer
        _ = self.llm_client
        self.summarizer.set_llm_client(self._llm_client)
        
        summaries = self.summarizer.summarize_all_communities(
            self.communities_info, self.chunks, graph, community_to_chunks
        )
        stats['num_summaries'] = len(summaries)
        
        # Compute embeddings
        logger.info("\nComputing embeddings...")
        self._compute_and_store_embeddings()
        
        # Save all data
        logger.info("\nSaving indices...")
        self._save_all()
        
        logger.info("\n" + "=" * 50)
        logger.info("Indexing Complete!")
        logger.info("=" * 50)
        logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
        
        return stats
    
    def _build_community_chunks_mapping(self) -> Dict[int, list]:
        """Build mapping from community ID to chunk IDs."""
        mapping = {}
        
        for comm_info in self.communities_info:
            comm_id = comm_info.get('id')
            chunk_ids = set()
            
            for entity in comm_info.get('entities', []):
                for cid in entity.get('chunk_ids', []):
                    chunk_ids.add(cid)
            
            mapping[comm_id] = list(chunk_ids)
        
        return mapping
    
    def _compute_and_store_embeddings(self):
        """Compute and store embeddings for chunks and entities."""
        # Chunk embeddings
        chunk_texts = [c.get('text', '') for c in self.chunks]
        self.chunk_embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Entity embeddings
        entity_texts = [e.get('normalized', e.get('text', '')) for e in self.entities]
        self.entity_embeddings = self.embedding_model.encode(
            entity_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Computed embeddings: {self.chunk_embeddings.shape[0]} chunks, "
                   f"{self.entity_embeddings.shape[0]} entities")
    
    def _save_all(self):
        """Save all indexed data to disk."""
        paths = self.config['paths']
        processed_dir = Path(paths['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        chunks_path = Path(paths['chunks_file'])
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved chunks to {chunks_path}")
        
        # Save entities
        entities_path = Path(paths['entities_file'])
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(self.entities, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved entities to {entities_path}")
        
        # Save knowledge graph
        graph_path = Path(paths['graph_file'])
        self.graph_builder.save(graph_path)
        
        # Save embeddings
        embeddings_path = Path(paths['embeddings_file'])
        np.savez(
            embeddings_path,
            chunk_embeddings=self.chunk_embeddings,
            entity_embeddings=self.entity_embeddings
        )
        logger.info(f"Saved embeddings to {embeddings_path}")
        
        # Save community summaries
        summaries_path = Path(paths['community_summaries_file'])
        self.summarizer.save_summaries(summaries_path)
        
        # Save communities info
        communities_path = processed_dir / 'communities.json'
        with open(communities_path, 'w', encoding='utf-8') as f:
            json.dump(self.communities_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved communities to {communities_path}")


def main():
    """Run the indexer from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SemRAG Indexer')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--pdf', help='Override PDF path')
    
    args = parser.parse_args()
    
    indexer = Indexer(config_path=args.config)
    indexer.run(pdf_path=args.pdf)


if __name__ == '__main__':
    main()
