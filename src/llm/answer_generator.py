"""
Answer Generator - Generates answers using LLM with retrieved context

Orchestrates the final step of the RAG pipeline.
"""

import logging
from typing import List, Dict, Optional, Generator

from .llm_client import OllamaClient
from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Generates answers using LLM with context from the retrieval system.
    
    Handles:
    - Context formatting
    - Prompt construction
    - Response generation
    - Citation tracking
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
        max_context_chunks: int = 5,
        include_citations: bool = True
    ):
        """
        Initialize answer generator.
        
        Args:
            llm_client: Ollama client instance
            max_context_chunks: Maximum chunks to include in context
            include_citations: Whether to track and include citations
        """
        self.llm_client = llm_client
        self.max_context_chunks = max_context_chunks
        self.include_citations = include_citations
        self.conversation_history = []
    
    def generate(
        self,
        query: str,
        retrieved_results: List[Dict],
        entities: Optional[List[str]] = None,
        community_summary: Optional[str] = None,
        stream: bool = False
    ):
        """
        Generate an answer for the query.
        
        Args:
            query: User query
            retrieved_results: Results from retrieval system
            entities: Relevant entities
            community_summary: Community context
            stream: Whether to stream the response
            
        Returns:
            Generated answer (string or generator for streaming)
        """
        logger.info(f"Generating answer for: '{query[:50]}...'")
        
        # Limit context chunks
        context_chunks = retrieved_results[:self.max_context_chunks]
        
        # Extract entities from results if not provided
        if entities is None:
            entities = self._extract_entities(context_chunks)
        
        # Extract community summary if not provided
        if community_summary is None:
            community_summary = self._extract_community_summary(context_chunks)
        
        # Build prompt
        prompt = PromptTemplates.format_qa_prompt(
            question=query,
            context_chunks=context_chunks,
            entities=entities,
            community_summary=community_summary
        )
        
        # Generate response
        if stream:
            return self._generate_stream(prompt)
        else:
            return self._generate_sync(prompt, context_chunks)
    
    def _generate_sync(
        self,
        prompt: str,
        context_chunks: List[Dict]
    ) -> Dict:
        """
        Generate synchronous response.
        
        Args:
            prompt: Formatted prompt
            context_chunks: Context for citations
            
        Returns:
            Response dict with answer and metadata
        """
        try:
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt=PromptTemplates.SYSTEM_PROMPT
            )
            
            result = {
                'answer': answer,
                'sources': self._format_sources(context_chunks) if self.include_citations else [],
                'entities': self._extract_entities(context_chunks),
                'success': True
            }
            
            # Update conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': prompt.split('QUESTION:')[-1].strip()
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': answer
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                'answer': f"I apologize, but I encountered an error generating the response: {str(e)}",
                'sources': [],
                'entities': [],
                'success': False,
                'error': str(e)
            }
    
    def _generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Generate streaming response.
        
        Args:
            prompt: Formatted prompt
            
        Yields:
            Response chunks
        """
        try:
            for chunk in self.llm_client.generate_stream(
                prompt=prompt,
                system_prompt=PromptTemplates.SYSTEM_PROMPT
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"\n\n[Error: {str(e)}]"
    
    def generate_with_history(
        self,
        query: str,
        retrieved_results: List[Dict]
    ) -> Dict:
        """
        Generate answer considering conversation history.
        
        Args:
            query: User query
            retrieved_results: Retrieved results
            
        Returns:
            Response dict
        """
        if not self.conversation_history:
            return self.generate(query, retrieved_results)
        
        # Format with history
        context_chunks = retrieved_results[:self.max_context_chunks]
        
        prompt = PromptTemplates.format_followup(
            question=query,
            history=self.conversation_history,
            context_chunks=context_chunks
        )
        
        return self._generate_sync(prompt, context_chunks)
    
    def _extract_entities(self, chunks: List[Dict]) -> List[str]:
        """Extract unique entities from chunks."""
        entities = set()
        for chunk in chunks:
            chunk_entities = chunk.get('entities', [])
            entities.update(chunk_entities)
        return list(entities)[:10]  # Limit
    
    def _extract_community_summary(self, chunks: List[Dict]) -> str:
        """Extract community summary from chunks."""
        for chunk in chunks:
            summary = chunk.get('community_summary', '')
            if summary:
                return summary
        return ""
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format chunks as citation sources."""
        sources = []
        for i, chunk in enumerate(chunks):
            sources.append({
                'id': i + 1,
                'chunk_id': chunk.get('chunk_id', ''),
                'text': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                'score': chunk.get('final_score', chunk.get('combined_score', 0)),
                'entities': chunk.get('entities', [])
            })
        return sources
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history.copy()


def create_answer_generator(
    model_name: str = "granite3-dense:2b",
    base_url: str = "http://localhost:11434"
) -> AnswerGenerator:
    """
    Factory function to create answer generator.
    
    Args:
        model_name: Ollama model name
        base_url: Ollama base URL
        
    Returns:
        Configured AnswerGenerator
    """
    llm_client = OllamaClient(
        model_name=model_name,
        base_url=base_url
    )
    
    return AnswerGenerator(llm_client=llm_client)
