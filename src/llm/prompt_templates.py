"""
Prompt Templates - Engineering prompts for SemRAG Q&A

Contains templates for various stages of the RAG pipeline.
"""

from typing import List, Dict, Optional


class PromptTemplates:
    """
    Collection of prompt templates for the SemRAG system.
    """
    
    # System prompt for Ambedkar Q&A
    SYSTEM_PROMPT = """You are AmbedkarGPT, an expert assistant specializing in the works and philosophy of Dr. B.R. Ambedkar. You have deep knowledge of:
- His writings on caste, untouchability, and social reform
- His role in drafting the Indian Constitution
- His views on Buddhism, education, and economics
- His political career and activism

Always:
- Provide accurate, well-sourced information
- Cite relevant sources when available
- Acknowledge when information is not available in the sources
- Be respectful and scholarly in tone
- Explain complex concepts clearly"""

    # Main Q&A template
    QA_TEMPLATE = """Based on the following retrieved information about Dr. B.R. Ambedkar's works, please answer the question.

RETRIEVED CONTEXT:
{context}

{entity_info}

{community_info}

QUESTION: {question}

Please provide a comprehensive, well-structured answer. If the information is not available in the context, acknowledge this limitation. Include specific references to the sources when making claims."""

    # Template with entities highlighted
    QA_WITH_ENTITIES_TEMPLATE = """Based on the following information, answer the question about Dr. B.R. Ambedkar.

RELEVANT ENTITIES:
{entities}

CONTEXT FROM SOURCES:
{context}

THEMATIC OVERVIEW:
{community_summary}

QUESTION: {question}

Provide a detailed answer that:
1. Addresses the question directly
2. References specific entities and their relationships
3. Uses evidence from the provided context
4. Acknowledges any gaps in available information"""

    # Community summarization template
    COMMUNITY_SUMMARY_TEMPLATE = """Analyze the following group of related entities and text excerpts to create a comprehensive summary.

ENTITIES IN THIS GROUP:
{entities}

RELATIONSHIPS:
{relationships}

RELEVANT TEXT:
{text_excerpts}

Create a 2-3 sentence summary that:
1. Identifies the main theme or topic
2. Highlights key entities and relationships
3. Captures the most important information

Summary:"""

    # Follow-up question template
    FOLLOWUP_TEMPLATE = """Previous conversation:
{history}

New question: {question}

RELEVANT CONTEXT:
{context}

Based on the conversation history and new context, provide a response that:
1. Addresses the new question
2. Maintains consistency with previous answers
3. Adds new relevant information from the context"""

    # Clarification request template
    CLARIFICATION_TEMPLATE = """The question "{question}" could be interpreted in multiple ways in the context of Dr. Ambedkar's works:

Possible interpretations:
{interpretations}

Could you please clarify which aspect you're interested in?"""

    @classmethod
    def format_qa_prompt(
        cls,
        question: str,
        context_chunks: List[Dict],
        entities: Optional[List[str]] = None,
        community_summary: Optional[str] = None
    ) -> str:
        """
        Format the main Q&A prompt.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            entities: Relevant entities
            community_summary: Community summary for context
            
        Returns:
            Formatted prompt string
        """
        # Format context
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            text = chunk.get('text', '')
            score = chunk.get('final_score', chunk.get('combined_score', 0))
            context_parts.append(f"[Source {i+1}] (relevance: {score:.2f})\n{text}")
        
        context = "\n\n".join(context_parts) if context_parts else "No specific context available."
        
        # Format entity info
        entity_info = ""
        if entities:
            entity_info = f"\nRELEVANT ENTITIES: {', '.join(entities[:10])}\n"
        
        # Format community info
        community_info = ""
        if community_summary:
            community_info = f"\nTHEMATIC CONTEXT: {community_summary}\n"
        
        return cls.QA_TEMPLATE.format(
            context=context,
            entity_info=entity_info,
            community_info=community_info,
            question=question
        )
    
    @classmethod
    def format_qa_with_entities(
        cls,
        question: str,
        context_chunks: List[Dict],
        entities: List[str],
        community_summary: str = ""
    ) -> str:
        """
        Format Q&A prompt with entity emphasis.
        
        Args:
            question: User question
            context_chunks: Retrieved chunks
            entities: Relevant entities
            community_summary: Community summary
            
        Returns:
            Formatted prompt
        """
        context = "\n\n".join([
            f"[{i+1}] {chunk.get('text', '')}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        return cls.QA_WITH_ENTITIES_TEMPLATE.format(
            entities=", ".join(entities) if entities else "None identified",
            context=context if context else "No specific context available.",
            community_summary=community_summary or "No thematic overview available.",
            question=question
        )
    
    @classmethod
    def format_followup(
        cls,
        question: str,
        history: List[Dict],
        context_chunks: List[Dict]
    ) -> str:
        """
        Format follow-up question prompt.
        
        Args:
            question: New question
            history: Conversation history
            context_chunks: Retrieved chunks
            
        Returns:
            Formatted prompt
        """
        # Format history
        history_parts = []
        for msg in history[-4:]:  # Last 4 messages
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            history_parts.append(f"{role.upper()}: {content}")
        
        history_str = "\n".join(history_parts) if history_parts else "No previous conversation."
        
        # Format context
        context = "\n\n".join([
            f"[{i+1}] {chunk.get('text', '')}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        return cls.FOLLOWUP_TEMPLATE.format(
            history=history_str,
            question=question,
            context=context if context else "No additional context."
        )
    
    @classmethod
    def format_community_summary(
        cls,
        entities: List[str],
        relationships: List[str],
        text_excerpts: List[str]
    ) -> str:
        """
        Format community summarization prompt.
        
        Args:
            entities: Entities in community
            relationships: Relationship descriptions
            text_excerpts: Text excerpts
            
        Returns:
            Formatted prompt
        """
        return cls.COMMUNITY_SUMMARY_TEMPLATE.format(
            entities=", ".join(entities),
            relationships="\n".join(f"- {r}" for r in relationships) if relationships else "None identified",
            text_excerpts="\n\n".join(f"- {t[:300]}..." for t in text_excerpts) if text_excerpts else "No excerpts available."
        )
