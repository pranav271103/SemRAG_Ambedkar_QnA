"""
Relationship Extractor - Extract relationships between entities

Uses dependency parsing and co-occurrence patterns to identify
relationships between entities.
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import spacy
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)


@dataclass
class Relationship:
    """Represents a relationship between two entities"""
    source: str           # Source entity (normalized)
    target: str           # Target entity (normalized)
    relation_type: str    # Type of relationship
    weight: float = 1.0   # Relationship strength
    chunk_ids: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)  # Sentences showing relationship
    
    def __hash__(self):
        return hash((self.source, self.target, self.relation_type))
    
    def __eq__(self, other):
        if isinstance(other, Relationship):
            return (self.source == other.source and 
                    self.target == other.target and
                    self.relation_type == other.relation_type)
        return False
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'relation_type': self.relation_type,
            'weight': self.weight,
            'chunk_ids': self.chunk_ids,
            'evidence': self.evidence[:3]  # Limit evidence stored
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Relationship':
        return cls(
            source=data['source'],
            target=data['target'],
            relation_type=data['relation_type'],
            weight=data.get('weight', 1.0),
            chunk_ids=data.get('chunk_ids', []),
            evidence=data.get('evidence', [])
        )


class RelationshipExtractor:
    """
    Extracts relationships between entities using:
    1. Dependency parsing (Subject-Verb-Object patterns)
    2. Co-occurrence within sentences
    3. Named relationship patterns
    """
    
    # Verb patterns indicating relationships
    RELATIONSHIP_VERBS = {
        'wrote': 'AUTHORED',
        'founded': 'FOUNDED',
        'led': 'LED',
        'created': 'CREATED',
        'established': 'ESTABLISHED',
        'drafted': 'DRAFTED',
        'criticized': 'CRITICIZED',
        'supported': 'SUPPORTED',
        'opposed': 'OPPOSED',
        'met': 'MET_WITH',
        'married': 'MARRIED',
        'born': 'BORN_IN',
        'died': 'DIED_IN',
        'studied': 'STUDIED_AT',
        'worked': 'WORKED_AT',
        'joined': 'JOINED',
        'resigned': 'RESIGNED_FROM',
        'converted': 'CONVERTED_TO',
        'advocated': 'ADVOCATED',
        'fought': 'FOUGHT_FOR',
        'belongs': 'BELONGS_TO',
        'represents': 'REPRESENTS',
        'influenced': 'INFLUENCED',
        'inspired': 'INSPIRED_BY',
    }
    
    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        cooccurrence_window: int = 2  # Sentences
    ):
        """
        Initialize relationship extractor.
        
        Args:
            model_name: spaCy model name
            cooccurrence_window: Window size for co-occurrence
        """
        logger.info(f"Loading spaCy model for relationship extraction: {model_name}")
        
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)
        
        self.cooccurrence_window = cooccurrence_window
        logger.info("Relationship extractor initialized")
    
    def _extract_svo_triples(self, doc: Doc) -> List[Tuple[str, str, str]]:
        """
        Extract Subject-Verb-Object triples from parsed document.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of (subject, verb, object) tuples
        """
        triples = []
        
        for token in doc:
            if token.pos_ == 'VERB':
                subjects = []
                objects = []
                
                # Find subjects
                for child in token.children:
                    if child.dep_ in ('nsubj', 'nsubjpass'):
                        # Get the full noun phrase
                        if child.pos_ in ('NOUN', 'PROPN', 'PRON'):
                            subjects.append(self._get_span_text(child))
                    elif child.dep_ in ('dobj', 'pobj', 'attr'):
                        if child.pos_ in ('NOUN', 'PROPN', 'PRON'):
                            objects.append(self._get_span_text(child))
                
                # Create triples
                for subj in subjects:
                    for obj in objects:
                        if subj and obj and subj != obj:
                            triples.append((subj, token.lemma_, obj))
        
        return triples
    
    def _get_span_text(self, token) -> str:
        """Get the text of a token including its compound parts."""
        parts = [token.text]
        
        for child in token.children:
            if child.dep_ == 'compound':
                parts.insert(0, child.text)
        
        return ' '.join(parts)
    
    def _extract_entity_cooccurrence(
        self,
        text: str,
        entities: Set[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Extract co-occurrence relationships between entities.
        
        Args:
            text: Input text
            entities: Set of entity names to look for
            
        Returns:
            List of (entity1, 'CO_OCCURS', entity2) tuples
        """
        doc = self.nlp(text)
        cooccurrences = []
        
        # Split into sentences
        sentences = list(doc.sents)
        
        for i, sent in enumerate(sentences):
            sent_text = sent.text.lower()
            
            # Find entities in this sentence and nearby sentences
            window_start = max(0, i - self.cooccurrence_window)
            window_end = min(len(sentences), i + self.cooccurrence_window + 1)
            
            window_text = ' '.join([sentences[j].text for j in range(window_start, window_end)])
            window_lower = window_text.lower()
            
            found_entities = [e for e in entities if e.lower() in window_lower]
            
            # Create co-occurrence pairs
            for j, e1 in enumerate(found_entities):
                for e2 in found_entities[j+1:]:
                    if e1 != e2:
                        cooccurrences.append((e1, 'CO_OCCURS', e2))
        
        return cooccurrences
    
    def extract_from_text(
        self,
        text: str,
        entities: Optional[Set[str]] = None
    ) -> List[Relationship]:
        """
        Extract relationships from text.
        
        Args:
            text: Input text
            entities: Optional set of entities to focus on
            
        Returns:
            List of relationships
        """
        doc = self.nlp(text)
        relationships = {}
        
        # 1. Extract SVO triples
        triples = self._extract_svo_triples(doc)
        
        for subj, verb, obj in triples:
            # Map verb to relationship type
            rel_type = self.RELATIONSHIP_VERBS.get(verb.lower(), 'RELATED_TO')
            
            key = (subj, obj, rel_type)
            if key not in relationships:
                relationships[key] = Relationship(
                    source=subj,
                    target=obj,
                    relation_type=rel_type,
                    evidence=[text[:200]]  # Store snippet
                )
            else:
                relationships[key].weight += 1
        
        # 2. Extract co-occurrence if entities provided
        if entities:
            cooc = self._extract_entity_cooccurrence(text, entities)
            for e1, rel_type, e2 in cooc:
                key = (e1, e2, rel_type)
                if key not in relationships:
                    relationships[key] = Relationship(
                        source=e1,
                        target=e2,
                        relation_type=rel_type
                    )
                else:
                    relationships[key].weight += 0.5  # Lower weight for co-occurrence
        
        return list(relationships.values())
    
    def extract_from_chunks(
        self,
        chunks: List[Dict],
        entities: List[Dict],
        min_weight: float = 0.5
    ) -> List[Relationship]:
        """
        Extract relationships from multiple chunks.
        
        Args:
            chunks: List of chunk dicts with 'id' and 'text'
            entities: List of entity dicts with 'normalized' key
            min_weight: Minimum relationship weight to include
            
        Returns:
            List of consolidated relationships
        """
        logger.info(f"Extracting relationships from {len(chunks)} chunks...")
        
        # Get entity names for co-occurrence
        entity_names = {e.get('normalized', e.get('text', '')) for e in entities}
        
        # Aggregate relationships
        all_relationships = {}
        
        for chunk in chunks:
            chunk_id = chunk.get('id', '')
            text = chunk.get('text', '')
            
            rels = self.extract_from_text(text, entity_names)
            
            for rel in rels:
                key = (rel.source, rel.target, rel.relation_type)
                
                if key not in all_relationships:
                    all_relationships[key] = rel
                    all_relationships[key].chunk_ids = [chunk_id]
                else:
                    all_relationships[key].weight += rel.weight
                    if chunk_id not in all_relationships[key].chunk_ids:
                        all_relationships[key].chunk_ids.append(chunk_id)
        
        # Filter by minimum weight
        filtered = [r for r in all_relationships.values() if r.weight >= min_weight]
        
        # Sort by weight
        filtered.sort(key=lambda x: x.weight, reverse=True)
        
        logger.info(f"Extracted {len(filtered)} unique relationships")
        
        return filtered
    
    def get_relationship_statistics(self, relationships: List[Relationship]) -> Dict:
        """
        Get statistics about extracted relationships.
        
        Args:
            relationships: List of relationships
            
        Returns:
            Dict with statistics
        """
        stats = {
            'total_relationships': len(relationships),
            'by_type': defaultdict(int),
            'avg_weight': 0,
            'top_relationships': []
        }
        
        for rel in relationships:
            stats['by_type'][rel.relation_type] += 1
        
        stats['by_type'] = dict(stats['by_type'])
        
        if relationships:
            stats['avg_weight'] = sum(r.weight for r in relationships) / len(relationships)
            stats['top_relationships'] = [
                {
                    'source': r.source,
                    'relation': r.relation_type,
                    'target': r.target,
                    'weight': r.weight
                }
                for r in relationships[:10]
            ]
        
        return stats
