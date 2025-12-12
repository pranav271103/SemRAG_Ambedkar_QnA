"""
Entity Extractor - Extract named entities from text chunks

Uses spaCy for Named Entity Recognition (NER) to identify
entities like PERSON, ORG, GPE, DATE, EVENT, etc.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re

import spacy
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    label: str  # Entity type (PERSON, ORG, GPE, etc.)
    normalized: str  # Normalized form
    frequency: int = 1
    chunk_ids: List[str] = field(default_factory=list)
    embedding: Optional[list] = None
    
    def __hash__(self):
        return hash((self.normalized, self.label))
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.normalized == other.normalized and self.label == other.label
        return False
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'label': self.label,
            'normalized': self.normalized,
            'frequency': self.frequency,
            'chunk_ids': self.chunk_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        return cls(
            text=data['text'],
            label=data['label'],
            normalized=data['normalized'],
            frequency=data.get('frequency', 1),
            chunk_ids=data.get('chunk_ids', [])
        )


class EntityExtractor:
    """
    Extracts named entities from text using spaCy.
    
    Supports custom entity rules for domain-specific terms.
    """
    
    # Entity types we care about
    RELEVANT_LABELS = {
        'PERSON',    # People, including fictional
        'ORG',       # Organizations
        'GPE',       # Geopolitical entities (countries, cities)
        'LOC',       # Non-GPE locations
        'DATE',      # Dates
        'EVENT',     # Named events
        'NORP',      # Nationalities, religious/political groups
        'FAC',       # Facilities (buildings, airports)
        'WORK_OF_ART',  # Titles of works
        'LAW',       # Named documents, laws
    }
    
    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        min_entity_length: int = 2,
        custom_entities: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the entity extractor.
        
        Args:
            model_name: spaCy model to use
            min_entity_length: Minimum character length for entities
            custom_entities: Dict mapping entity text to label
        """
        logger.info(f"Loading spaCy model: {model_name}")
        
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model {model_name} not found. Downloading...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)
        
        self.min_entity_length = min_entity_length
        self.custom_entities = custom_entities or {}
        
        # Add custom entity patterns for Ambedkar-related terms
        self._add_ambedkar_patterns()
        
        logger.info("Entity extractor initialized")
    
    def _add_ambedkar_patterns(self):
        """Add custom patterns for Ambedkar-related entities."""
        # Common entities in Ambedkar's works
        ambedkar_entities = {
            "Dr. B.R. Ambedkar": "PERSON",
            "Dr. Ambedkar": "PERSON",
            "B.R. Ambedkar": "PERSON",
            "Bhimrao Ambedkar": "PERSON",
            "Babasaheb": "PERSON",
            "Mahatma Gandhi": "PERSON",
            "Gandhi": "PERSON",
            "Jawaharlal Nehru": "PERSON",
            "Constituent Assembly": "ORG",
            "Indian National Congress": "ORG",
            "Congress": "ORG",
            "Scheduled Castes Federation": "ORG",
            "Republican Party of India": "ORG",
            "Hindu Code Bill": "LAW",
            "Indian Constitution": "LAW",
            "Poona Pact": "EVENT",
            "Round Table Conference": "EVENT",
            "Mahad Satyagraha": "EVENT",
            "Temple Entry Movement": "EVENT",
            "Dalit": "NORP",
            "Untouchables": "NORP",
            "Dalits": "NORP",
            "Brahmins": "NORP",
            "Buddhism": "NORP",
        }
        
        self.custom_entities.update(ambedkar_entities)
    
    def _normalize_entity(self, text: str) -> str:
        """
        Normalize entity text for deduplication.
        
        Args:
            text: Entity text
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common prefixes
        prefixes = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'The']
        for prefix in prefixes:
            if text.startswith(prefix + ' '):
                text = text[len(prefix):].strip()
        
        return text.strip()
    
    def _apply_custom_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Find custom entities in text.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity_text, label, start, end) tuples
        """
        found = []
        text_lower = text.lower()
        
        for entity_text, label in self.custom_entities.items():
            entity_lower = entity_text.lower()
            start = 0
            while True:
                pos = text_lower.find(entity_lower, start)
                if pos == -1:
                    break
                found.append((entity_text, label, pos, pos + len(entity_text)))
                start = pos + 1
        
        return found
    
    def extract_from_text(self, text: str) -> List[Entity]:
        """
        Extract entities from a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        doc = self.nlp(text)
        entities = {}
        
        # Extract spaCy entities
        for ent in doc.ents:
            if ent.label_ not in self.RELEVANT_LABELS:
                continue
            
            if len(ent.text) < self.min_entity_length:
                continue
            
            normalized = self._normalize_entity(ent.text)
            if not normalized:
                continue
            
            key = (normalized, ent.label_)
            if key not in entities:
                entities[key] = Entity(
                    text=ent.text,
                    label=ent.label_,
                    normalized=normalized
                )
            else:
                entities[key].frequency += 1
        
        # Add custom entities
        custom = self._apply_custom_entities(text)
        for ent_text, label, _, _ in custom:
            normalized = self._normalize_entity(ent_text)
            key = (normalized, label)
            if key not in entities:
                entities[key] = Entity(
                    text=ent_text,
                    label=label,
                    normalized=normalized
                )
            else:
                entities[key].frequency += 1
        
        return list(entities.values())
    
    def extract_from_chunks(
        self,
        chunks: List[Dict],
        min_frequency: int = 1
    ) -> Tuple[List[Entity], Dict[str, List[str]]]:
        """
        Extract entities from multiple chunks.
        
        Args:
            chunks: List of chunk dicts with 'id' and 'text' keys
            min_frequency: Minimum frequency for entity inclusion
            
        Returns:
            Tuple of (entities, entity_to_chunks mapping)
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks...")
        
        # Aggregate entities across all chunks
        all_entities = {}  # (normalized, label) -> Entity
        entity_to_chunks = defaultdict(list)
        
        for chunk in chunks:
            chunk_id = chunk.get('id', str(hash(chunk.get('text', ''))))
            text = chunk.get('text', '')
            
            entities = self.extract_from_text(text)
            
            for entity in entities:
                key = (entity.normalized, entity.label)
                
                if key not in all_entities:
                    all_entities[key] = entity
                    all_entities[key].chunk_ids = [chunk_id]
                else:
                    all_entities[key].frequency += entity.frequency
                    if chunk_id not in all_entities[key].chunk_ids:
                        all_entities[key].chunk_ids.append(chunk_id)
                
                entity_to_chunks[entity.normalized].append(chunk_id)
        
        # Filter by minimum frequency
        filtered_entities = [
            e for e in all_entities.values()
            if e.frequency >= min_frequency
        ]
        
        # Sort by frequency
        filtered_entities.sort(key=lambda x: x.frequency, reverse=True)
        
        logger.info(f"Extracted {len(filtered_entities)} unique entities")
        
        return filtered_entities, dict(entity_to_chunks)
    
    def get_entity_statistics(self, entities: List[Entity]) -> Dict:
        """
        Get statistics about extracted entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Dict with statistics
        """
        stats = {
            'total_entities': len(entities),
            'by_type': defaultdict(int),
            'top_entities': [],
            'avg_frequency': 0
        }
        
        for entity in entities:
            stats['by_type'][entity.label] += 1
        
        stats['by_type'] = dict(stats['by_type'])
        
        if entities:
            stats['avg_frequency'] = sum(e.frequency for e in entities) / len(entities)
            stats['top_entities'] = [
                {'text': e.text, 'label': e.label, 'frequency': e.frequency}
                for e in entities[:10]
            ]
        
        return stats
