"""
Semantic Chunker - Algorithm 1 from SemRAG Paper

This module implements semantic chunking by:
1. Splitting text into sentences
2. Encoding sentences with embedding model
3. Calculating cosine similarity between consecutive sentences
4. Grouping sentences where similarity > threshold
5. Applying buffer merging for context preservation
6. Enforcing token limits
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a semantic chunk of text"""
    id: str
    text: str
    sentences: List[str]
    start_idx: int  # Starting sentence index in original document
    end_idx: int    # Ending sentence index in original document
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    
    def __len__(self):
        return len(self.text.split())
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'sentences': self.sentences,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Chunk':
        return cls(
            id=data['id'],
            text=data['text'],
            sentences=data['sentences'],
            start_idx=data['start_idx'],
            end_idx=data['end_idx'],
            metadata=data.get('metadata', {})
        )


class SemanticChunker:
    """
    Implements Algorithm 1 from SemRAG paper for semantic chunking.
    
    The algorithm groups sentences based on semantic similarity using
    sentence embeddings and cosine similarity.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        max_tokens: int = 1024,
        min_chunk_size: int = 50,
        buffer_size: int = 2,
        device: str = "cpu"
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            model_name: Name of the sentence transformer model
            similarity_threshold: Threshold for grouping sentences (τ)
            max_tokens: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk
            buffer_size: Number of sentences for buffer merging
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self.min_chunk_size = min_chunk_size
        self.buffer_size = buffer_size
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info("Embedding model loaded successfully")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Use NLTK sentence tokenizer
        sentences = nltk.sent_tokenize(text)
        
        # Filter out very short sentences (likely artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Encode sentences into embeddings.
        
        Args:
            sentences: List of sentences to encode
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(
            sentences,
            show_progress_bar=len(sentences) > 100,
            convert_to_numpy=True
        )
        return embeddings
    
    def _calculate_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between consecutive sentence embeddings.
        
        Args:
            embeddings: Array of sentence embeddings
            
        Returns:
            Array of similarity scores between consecutive sentences
        """
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        return np.array(similarities)
    
    def _group_sentences(
        self,
        sentences: List[str],
        similarities: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Group sentences based on similarity threshold.
        
        Algorithm 1 Core Logic:
        - If similarity between consecutive sentences >= threshold, group them
        - Otherwise, start a new group
        
        Args:
            sentences: List of sentences
            similarities: Similarity scores between consecutive sentences
            
        Returns:
            List of (start_idx, end_idx) tuples for each group
        """
        if len(sentences) == 0:
            return []
        
        if len(sentences) == 1:
            return [(0, 0)]
        
        groups = []
        current_start = 0
        
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                # End current group and start new one
                groups.append((current_start, i))
                current_start = i + 1
        
        # Don't forget the last group
        groups.append((current_start, len(sentences) - 1))
        
        return groups
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.
        Simple approximation: ~1.3 tokens per word
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        words = len(text.split())
        return int(words * 1.3)
    
    def _merge_small_chunks(
        self,
        chunks: List[Chunk],
        sentences: List[str]
    ) -> List[Chunk]:
        """
        Merge chunks that are too small (buffer merging).
        
        Args:
            chunks: List of initial chunks
            sentences: Original sentences for reference
            
        Returns:
            List of merged chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            current_tokens = self._estimate_tokens(current_chunk.text)
            next_tokens = self._estimate_tokens(next_chunk.text)
            
            # If current chunk is too small, merge with next
            if current_tokens < self.min_chunk_size:
                # Merge chunks
                merged_text = current_chunk.text + " " + next_chunk.text
                merged_sentences = current_chunk.sentences + next_chunk.sentences
                current_chunk = Chunk(
                    id=current_chunk.id,
                    text=merged_text,
                    sentences=merged_sentences,
                    start_idx=current_chunk.start_idx,
                    end_idx=next_chunk.end_idx
                )
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk
        
        merged.append(current_chunk)
        return merged
    
    def _split_large_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Split chunks that exceed max token limit.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of chunks with large ones split
        """
        result = []
        
        for chunk in chunks:
            tokens = self._estimate_tokens(chunk.text)
            
            if tokens <= self.max_tokens:
                result.append(chunk)
            else:
                # Split into smaller sub-chunks
                sentences = chunk.sentences
                current_sentences = []
                current_tokens = 0
                sub_chunk_idx = 0
                
                for i, sentence in enumerate(sentences):
                    sent_tokens = self._estimate_tokens(sentence)
                    
                    if current_tokens + sent_tokens > self.max_tokens and current_sentences:
                        # Create sub-chunk
                        sub_chunk = Chunk(
                            id=f"{chunk.id}_sub{sub_chunk_idx}",
                            text=" ".join(current_sentences),
                            sentences=current_sentences.copy(),
                            start_idx=chunk.start_idx + (i - len(current_sentences)),
                            end_idx=chunk.start_idx + i - 1
                        )
                        result.append(sub_chunk)
                        current_sentences = [sentence]
                        current_tokens = sent_tokens
                        sub_chunk_idx += 1
                    else:
                        current_sentences.append(sentence)
                        current_tokens += sent_tokens
                
                # Don't forget remaining sentences
                if current_sentences:
                    sub_chunk = Chunk(
                        id=f"{chunk.id}_sub{sub_chunk_idx}",
                        text=" ".join(current_sentences),
                        sentences=current_sentences,
                        start_idx=chunk.start_idx + (len(sentences) - len(current_sentences)),
                        end_idx=chunk.end_idx
                    )
                    result.append(sub_chunk)
        
        return result
    
    def chunk(self, text: str, doc_id: str = "doc") -> List[Chunk]:
        """
        Main chunking method implementing Algorithm 1 from SemRAG.
        
        Args:
            text: Input text to chunk
            doc_id: Document identifier for chunk IDs
            
        Returns:
            List of semantic chunks
        """
        logger.info("Starting semantic chunking...")
        
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(text)
        logger.info(f"Split into {len(sentences)} sentences")
        
        if len(sentences) == 0:
            return []
        
        if len(sentences) == 1:
            return [Chunk(
                id=f"{doc_id}_chunk_0",
                text=sentences[0],
                sentences=sentences,
                start_idx=0,
                end_idx=0
            )]
        
        # Step 2: Encode sentences
        logger.info("Encoding sentences...")
        embeddings = self._encode_sentences(sentences)
        
        # Step 3: Calculate similarities between consecutive sentences
        logger.info("Calculating similarities...")
        similarities = self._calculate_similarities(embeddings)
        
        # Log similarity statistics
        logger.info(f"Similarity stats - Mean: {similarities.mean():.3f}, "
                   f"Std: {similarities.std():.3f}, "
                   f"Min: {similarities.min():.3f}, "
                   f"Max: {similarities.max():.3f}")
        
        # Step 4: Group sentences based on similarity threshold
        groups = self._group_sentences(sentences, similarities)
        logger.info(f"Created {len(groups)} initial groups")
        
        # Step 5: Create chunk objects
        chunks = []
        for idx, (start, end) in enumerate(groups):
            chunk_sentences = sentences[start:end + 1]
            chunk = Chunk(
                id=f"{doc_id}_chunk_{idx}",
                text=" ".join(chunk_sentences),
                sentences=chunk_sentences,
                start_idx=start,
                end_idx=end
            )
            chunks.append(chunk)
        
        # Step 6: Buffer merging for small chunks
        chunks = self._merge_small_chunks(chunks, sentences)
        logger.info(f"After buffer merging: {len(chunks)} chunks")
        
        # Step 7: Split large chunks
        chunks = self._split_large_chunks(chunks)
        logger.info(f"After splitting large chunks: {len(chunks)} chunks")
        
        # Step 8: Re-assign IDs after merging/splitting
        for idx, chunk in enumerate(chunks):
            chunk.id = f"{doc_id}_chunk_{idx}"
        
        # Step 9: Compute embeddings for final chunks
        logger.info("Computing chunk embeddings...")
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = self._encode_sentences(chunk_texts)
        for i, chunk in enumerate(chunks):
            chunk.embedding = chunk_embeddings[i]
        
        logger.info(f"Semantic chunking complete. Created {len(chunks)} chunks.")
        return chunks
    
    def chunk_with_overlap(
        self,
        text: str,
        doc_id: str = "doc",
        overlap_sentences: int = 1
    ) -> List[Chunk]:
        """
        Chunk with overlap between consecutive chunks for better context.
        
        Args:
            text: Input text to chunk
            doc_id: Document identifier
            overlap_sentences: Number of sentences to overlap
            
        Returns:
            List of chunks with overlap
        """
        # First, get regular chunks
        chunks = self.chunk(text, doc_id)
        
        if len(chunks) <= 1 or overlap_sentences == 0:
            return chunks
        
        # Add overlap by including last sentences from previous chunk
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Get overlap sentences from previous chunk
            overlap_sents = prev_chunk.sentences[-overlap_sentences:]
            
            # Create new chunk with overlap
            new_sentences = overlap_sents + curr_chunk.sentences
            new_chunk = Chunk(
                id=f"{doc_id}_chunk_{i}",
                text=" ".join(new_sentences),
                sentences=new_sentences,
                start_idx=prev_chunk.end_idx - overlap_sentences + 1,
                end_idx=curr_chunk.end_idx,
                metadata={'has_overlap': True}
            )
            overlapped_chunks.append(new_chunk)
        
        # Recompute embeddings
        chunk_texts = [c.text for c in overlapped_chunks]
        chunk_embeddings = self._encode_sentences(chunk_texts)
        for i, chunk in enumerate(overlapped_chunks):
            chunk.embedding = chunk_embeddings[i]
        
        return overlapped_chunks


def semantic_chunk(
    text: str,
    threshold: float = 0.5,
    max_tokens: int = 1024,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[Chunk]:
    """
    Convenience function for semantic chunking.
    
    Args:
        text: Input text
        threshold: Similarity threshold
        max_tokens: Maximum tokens per chunk
        model_name: Embedding model name
        
    Returns:
        List of semantic chunks
    """
    chunker = SemanticChunker(
        model_name=model_name,
        similarity_threshold=threshold,
        max_tokens=max_tokens
    )
    return chunker.chunk(text)
