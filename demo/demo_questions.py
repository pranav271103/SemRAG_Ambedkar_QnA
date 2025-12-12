"""
Demo Script for AmbedkarGPT

Run this script to demonstrate the SemRAG Q&A system
with sample questions about Dr. B.R. Ambedkar's works.
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header():
    """Print demo header."""
    print("\n" + "=" * 60)
    print("🏛️  AmbedkarGPT - SemRAG Q&A Demo")
    print("=" * 60)
    print("Demonstrating the SemRAG-based Q&A system on")
    print("Dr. B.R. Ambedkar's works\n")


def print_question(q: str, num: int):
    """Print formatted question."""
    print(f"\n{'─' * 60}")
    print(f"📝 Question {num}: {q}")
    print("─" * 60)


def print_answer(response: dict, elapsed: float):
    """Print formatted answer."""
    answer = response.get('answer', 'No answer generated.')
    sources = response.get('sources', [])
    entities = response.get('entities', [])
    
    print(f"\n🤖 Answer:\n{answer}")
    
    if entities:
        print(f"\n🏷️  Relevant Entities: {', '.join(entities[:5])}")
    
    if sources:
        print(f"\n📚 Sources ({len(sources)}):")
        for i, source in enumerate(sources[:3]):
            text = source.get('text', '')[:150]
            score = source.get('score', 0)
            print(f"   [{i+1}] (score: {score:.2f}) {text}...")
    
    print(f"\n⏱️  Response time: {elapsed:.2f}s")


def run_demo():
    """Run the demo with sample questions."""
    print_header()
    
    # Sample questions
    questions = [
        "What were Dr. Ambedkar's views on the caste system?",
        "What role did Ambedkar play in drafting the Indian Constitution?",
        "Explain Ambedkar's thoughts on education and social reform.",
        "Describe Ambedkar's conversion to Buddhism.",
        "What was the Poona Pact and its significance?"
    ]
    
    print("Loading AmbedkarGPT...")
    
    try:
        from src.pipeline.query_engine import QueryEngine
        
        engine = QueryEngine(config_path="config.yaml")
        
        if not engine.load_indices():
            print("\n❌ Error: Indices not found!")
            print("Please run the indexer first:")
            print("  python -m src.pipeline.indexer")
            return
        
        stats = engine.get_statistics()
        print(f"\n✅ Loaded: {stats['num_chunks']} chunks, "
              f"{stats['num_entities']} entities, "
              f"{stats['num_communities']} communities")
        
        print("\n" + "=" * 60)
        print("Starting Q&A Demo...")
        print("=" * 60)
        
        for i, question in enumerate(questions, 1):
            print_question(question, i)
            
            start_time = time.time()
            response = engine.query(
                question=question,
                search_type="hybrid",
                top_k=5
            )
            elapsed = time.time() - start_time
            
            print_answer(response, elapsed)
            
            # Small pause between questions
            if i < len(questions):
                print("\n" + "." * 60)
                time.sleep(1)
        
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        
        # Interactive mode
        print("\n💬 Try your own questions (type 'quit' to exit):\n")
        
        while True:
            try:
                user_q = input("Your question: ").strip()
                
                if not user_q:
                    continue
                
                if user_q.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋")
                    break
                
                start_time = time.time()
                response = engine.query(user_q, search_type="hybrid")
                elapsed = time.time() - start_time
                
                print_answer(response, elapsed)
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    run_demo()
