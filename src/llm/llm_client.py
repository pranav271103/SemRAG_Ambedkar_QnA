"""
LLM Client - Ollama integration for local LLM inference

Provides a unified interface for interacting with Ollama models.
"""

import logging
from typing import Optional, Dict, Generator, List
import json

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama local LLM.
    
    Supports both streaming and non-streaming generation.
    """
    
    def __init__(
        self,
        model_name: str = "granite3-dense:2b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Name of the Ollama model
            base_url: Ollama API base URL
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to Ollama server."""
        try:
            import ollama
            
            # Try to list models
            models = ollama.list()
            available_models = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
            
            logger.info(f"Connected to Ollama. Available models: {available_models}")
            
            # Check if our model is available
            model_found = any(self.model_name in m for m in available_models)
            if not model_found:
                logger.warning(f"Model '{self.model_name}' not found. "
                             f"Available: {available_models}")
            else:
                logger.info(f"Using model: {self.model_name}")
                
        except ImportError:
            logger.error("ollama package not installed. Run: pip install ollama")
            raise
        except Exception as e:
            logger.warning(f"Could not verify Ollama connection: {e}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text response
        """
        import ollama
        
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': temp,
                    'num_predict': max_tok
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            
        Yields:
            Text chunks as they are generated
        """
        import ollama
        
        temp = temperature if temperature is not None else self.temperature
        
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={'temperature': temp},
                stream=True
            )
            
            for chunk in stream:
                content = chunk.get('message', {}).get('content', '')
                if content:
                    yield content
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response with context from retrieved documents.
        
        Args:
            query: User query
            context: List of context strings
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        # Build context string
        context_str = "\n\n".join([
            f"[Source {i+1}]: {ctx}"
            for i, ctx in enumerate(context)
        ])
        
        prompt = f"""Based on the following sources, answer the question.

SOURCES:
{context_str}

QUESTION: {query}

Provide a comprehensive answer based on the sources above. Include relevant details and cite sources when appropriate."""
        
        return self.generate(prompt, system_prompt)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None
    ) -> str:
        """
        Multi-turn chat with message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            
        Returns:
            Assistant response
        """
        import ollama
        
        temp = temperature if temperature is not None else self.temperature
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={'temperature': temp}
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using Ollama's embedding API.
        
        Note: Not all models support embeddings. Use sentence-transformers
        for embeddings in the main pipeline.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list
        """
        import ollama
        
        try:
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.warning(f"Embedding failed (model may not support embeddings): {e}")
            return []
