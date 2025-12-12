"""
Streamlit UI for AmbedkarGPT

Beautiful, interactive interface for the SemRAG Q&A system.
Features:
- Chat interface with conversation history
- Knowledge graph visualization (separate tab)
- Source citations with highlighting
- Search type selection
- Performance metrics
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_custom_css():
    """Load custom CSS for styling."""
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        color: white;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Source card styling */
    .source-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Entity tag styling */
    .entity-tag {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: transparent;
        border-radius: 10px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    
    if 'indices_loaded' not in st.session_state:
        st.session_state.indices_loaded = False
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []


def load_query_engine():
    """Load the query engine."""
    if st.session_state.query_engine is None:
        with st.spinner("🔄 Loading AmbedkarGPT..."):
            try:
                from src.pipeline.query_engine import QueryEngine
                
                engine = QueryEngine(config_path="config.yaml")
                success = engine.load_indices()
                
                if success:
                    st.session_state.query_engine = engine
                    st.session_state.indices_loaded = True
                    st.success("✅ AmbedkarGPT loaded successfully!")
                else:
                    st.error("❌ Failed to load indices. Please run the indexer first.")
                    st.code("python -m src.pipeline.indexer", language="bash")
                    return False
                    
            except Exception as e:
                st.error(f"❌ Error loading system: {str(e)}")
                return False
    
    return st.session_state.indices_loaded


def render_header():
    """Render the application header."""
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   font-size: 3rem;
                   margin-bottom: 0;">
            🏛️ AmbedkarGPT
        </h1>
        <p style="color: #a0aec0; font-size: 1.1rem;">
            SemRAG-powered Q&A on Dr. B.R. Ambedkar's Works
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with settings and info."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Search type selection
        search_type = st.selectbox(
            "Search Method",
            ["hybrid", "local", "global"],
            help="Hybrid combines local and global search for best results"
        )
        
        # Number of results
        top_k = st.slider("Number of Sources", 3, 10, 5)
        
        st.markdown("---")
        
        # Statistics
        if st.session_state.indices_loaded and st.session_state.query_engine:
            stats = st.session_state.query_engine.get_statistics()
            
            st.markdown("## 📊 System Stats")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chunks", stats.get('num_chunks', 0))
                st.metric("Entities", stats.get('num_entities', 0))
            with col2:
                st.metric("Communities", stats.get('num_communities', 0))
                st.metric("Graph Edges", stats.get('graph_edges', 0))
        
        st.markdown("---")
        
        # Clear history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if st.session_state.query_engine:
                st.session_state.query_engine.clear_history()
            st.rerun()
        
        return search_type, top_k


def get_sample_questions():
    """Return sample questions."""
    return [
        "What were Ambedkar's views on the caste system?",
        "How did Ambedkar contribute to the Constitution?",
        "What was Ambedkar's educational philosophy?",
        "Describe Ambedkar's conversion to Buddhism.",
        "What was the Poona Pact?"
    ]


def render_chat_message(message: dict):
    """Render a chat message."""
    role = message.get('role', 'user')
    content = message.get('content', '')
    
    if role == 'user':
        st.markdown(f"""
        <div class="user-message">
            <strong>🙋 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🏛️ AmbedkarGPT:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)


def render_sources(sources: list):
    """Render source citations."""
    if not sources:
        return
    
    with st.expander("📚 Sources", expanded=False):
        for i, source in enumerate(sources):
            score = source.get('score', source.get('final_score', 0))
            text = source.get('text', '')
            entities = source.get('entities', [])
            
            st.markdown(f"""
            <div class="source-card">
                <strong>Source {i+1}</strong> 
                <span style="color: #667eea;">(relevance: {score:.2f})</span>
                <p style="margin-top: 0.5rem; color: #a0aec0;">{text[:300]}...</p>
            </div>
            """, unsafe_allow_html=True)
            
            if entities:
                entity_html = " ".join([
                    f'<span class="entity-tag">{e}</span>'
                    for e in entities[:5]
                ])
                st.markdown(entity_html, unsafe_allow_html=True)


def create_knowledge_graph_html(graph_builder, max_nodes=100, filter_type=None, search_term=None):
    """
    Create an interactive knowledge graph visualization.
    
    Returns HTML string for the graph.
    """
    try:
        from pyvis.network import Network
        import networkx as nx
    except ImportError:
        return None, "PyVis not installed. Run: pip install pyvis"
    
    if not graph_builder or not graph_builder.graph:
        return None, "Knowledge graph not loaded"
    
    graph = graph_builder.graph
    
    # Filter nodes if search term provided
    if search_term:
        matching_nodes = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            text = node_data.get('text', '').lower()
            if search_term.lower() in text:
                matching_nodes.append(node)
                # Also add neighbors
                for neighbor in graph.neighbors(node):
                    if neighbor not in matching_nodes:
                        matching_nodes.append(neighbor)
        nodes_to_show = matching_nodes[:max_nodes]
    else:
        # Get nodes sorted by degree (most connected first)
        node_degrees = dict(graph.degree())
        sorted_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)
        nodes_to_show = sorted_nodes[:max_nodes]
    
    # Filter by type if specified
    if filter_type and filter_type != "All":
        nodes_to_show = [
            n for n in nodes_to_show 
            if graph.nodes[n].get('label', '') == filter_type
        ]
    
    if not nodes_to_show:
        return None, "No nodes match the filter criteria"
    
    subgraph = graph.subgraph(nodes_to_show)
    
    # Create pyvis network
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=False
    )
    
    # Physics settings for better layout
    net.set_options("""
    {
        "nodes": {
            "font": {"size": 14, "face": "arial"},
            "scaling": {"min": 10, "max": 50}
        },
        "edges": {
            "color": {"inherit": true},
            "smooth": {"type": "continuous"}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
        }
    }
    """)
    
    # Color map for entity types
    color_map = {
        'PERSON': '#f093fb',
        'ORG': '#667eea',
        'GPE': '#48bb78',
        'LOC': '#38b2ac',
        'EVENT': '#f5576c',
        'DATE': '#fbd38d',
        'LAW': '#9f7aea',
        'NORP': '#ed8936',
        'FAC': '#4fd1c5',
        'WORK_OF_ART': '#fc8181',
        'UNKNOWN': '#a0aec0'
    }
    
    # Add nodes
    for node in subgraph.nodes():
        node_data = graph.nodes[node]
        label = node_data.get('text', node)
        node_type = node_data.get('label', 'UNKNOWN')
        frequency = node_data.get('frequency', 1)
        color = color_map.get(node_type, '#a0aec0')
        
        # Size based on frequency and connections
        degree = graph.degree(node)
        size = min(15 + frequency * 3 + degree * 2, 60)
        
        # Create tooltip with details
        chunk_count = len(node_data.get('chunk_ids', []))
        title = f"""<b>{label}</b><br>
Type: {node_type}<br>
Frequency: {frequency}<br>
Connections: {degree}<br>
Mentioned in {chunk_count} chunks"""
        
        net.add_node(
            node,
            label=label[:25] + "..." if len(label) > 25 else label,
            color=color,
            size=size,
            title=title,
            borderWidth=2,
            borderWidthSelected=4
        )
    
    # Add edges
    for source, target in subgraph.edges():
        edge_data = graph[source][target]
        weight = edge_data.get('weight', 1)
        rel_types = edge_data.get('relation_types', ['RELATED'])
        
        # Edge thickness based on weight
        width = min(1 + weight * 0.5, 5)
        
        title = f"Relationship: {', '.join(rel_types[:2])}<br>Weight: {weight}"
        
        net.add_edge(
            source,
            target,
            width=width,
            title=title
        )
    
    # Generate HTML
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
            net.save_graph(f.name)
            with open(f.name, 'r', encoding='utf-8') as html_file:
                html_content = html_file.read()
            return html_content, None
    except Exception as e:
        return None, f"Error generating graph: {str(e)}"


def render_knowledge_graph_tab():
    """Render the Knowledge Graph tab with full interactive visualization."""
    st.markdown("## 🕸️ Interactive Knowledge Graph")
    st.markdown("Explore the relationships between entities in Dr. Ambedkar's works.")
    
    if not st.session_state.indices_loaded:
        st.warning("Please load the system first from the Chat tab.")
        return
    
    engine = st.session_state.query_engine
    if not engine or not engine._graph_builder:
        st.warning("Knowledge graph not available.")
        return
    
    graph_builder = engine.graph_builder
    graph = graph_builder.graph
    
    # Graph controls
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        max_nodes = st.slider("Max Nodes to Display", 20, 200, 80, 10)
    
    with col2:
        # Get available entity types
        entity_types = set()
        for node in graph.nodes():
            entity_types.add(graph.nodes[node].get('label', 'UNKNOWN'))
        type_options = ["All"] + sorted(list(entity_types))
        filter_type = st.selectbox("Filter by Type", type_options)
    
    with col3:
        search_term = st.text_input("Search Entity", placeholder="e.g., Ambedkar")
    
    # Legend
    st.markdown("### 🎨 Entity Types Legend")
    legend_cols = st.columns(6)
    color_map = {
        'PERSON': '#f093fb',
        'ORG': '#667eea',
        'GPE': '#48bb78',
        'EVENT': '#f5576c',
        'LAW': '#9f7aea',
        'NORP': '#ed8936'
    }
    for i, (etype, color) in enumerate(color_map.items()):
        with legend_cols[i % 6]:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 16px; height: 16px; background: {color}; border-radius: 50%;"></div>
                <span style="font-size: 0.9rem;">{etype}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create and display graph
    with st.spinner("Generating interactive graph..."):
        html_content, error = create_knowledge_graph_html(
            graph_builder,
            max_nodes=max_nodes,
            filter_type=filter_type if filter_type != "All" else None,
            search_term=search_term if search_term else None
        )
    
    if error:
        st.error(error)
        if "pyvis" in error.lower():
            st.code("pip install pyvis", language="bash")
    elif html_content:
        # Display the graph
        st.components.v1.html(html_content, height=720, scrolling=False)
        
        # Graph statistics
        st.markdown("### 📊 Graph Statistics")
        stats_cols = st.columns(4)
        with stats_cols[0]:
            st.metric("Total Nodes", graph.number_of_nodes())
        with stats_cols[1]:
            st.metric("Total Edges", graph.number_of_edges())
        with stats_cols[2]:
            st.metric("Displayed Nodes", min(max_nodes, graph.number_of_nodes()))
        with stats_cols[3]:
            avg_degree = sum(dict(graph.degree()).values()) / max(graph.number_of_nodes(), 1)
            st.metric("Avg Connections", f"{avg_degree:.1f}")
        
        # Top entities
        st.markdown("### 🏆 Most Connected Entities")
        node_degrees = dict(graph.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for node_id, degree in top_nodes:
            node_data = graph.nodes[node_id]
            text = node_data.get('text', node_id)
            label = node_data.get('label', 'UNKNOWN')
            st.markdown(f"**{text}** ({label}) - {degree} connections")


def render_chat_tab(search_type, top_k):
    """Render the Chat tab."""
    st.markdown("### 💬 Chat with AmbedkarGPT")
    
    # Sample questions
    st.markdown("**💡 Try a sample question:**")
    sample_cols = st.columns(3)
    sample_questions = get_sample_questions()
    selected_sample = None
    
    for i, q in enumerate(sample_questions[:3]):
        with sample_cols[i]:
            if st.button(q[:40] + "...", key=f"sample_{i}"):
                selected_sample = q
    
    # Display chat history
    for message in st.session_state.messages:
        render_chat_message(message)
    
    # Sources from last query
    if st.session_state.search_results:
        render_sources(st.session_state.search_results)
    
    # Chat input
    question = st.chat_input("Ask about Dr. B.R. Ambedkar's works...")
    
    if selected_sample:
        question = selected_sample
    
    if question:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': question
        })
        
        # Process query
        with st.spinner("🔍 Searching and generating answer..."):
            start_time = time.time()
            response = st.session_state.query_engine.query(
                question=question,
                search_type=search_type,
                top_k=top_k
            )
            elapsed_time = time.time() - start_time
        
        if response.get('success', True):
            answer = response.get('answer', 'No answer generated.')
            sources = response.get('sources', [])
            
            # Add assistant message
            st.session_state.messages.append({
                'role': 'assistant',
                'content': answer
            })
            
            st.session_state.search_results = sources
            st.toast(f"Response generated in {elapsed_time:.2f}s")
            
        else:
            st.error(f"Error: {response.get('error', 'Unknown error')}")
        
        st.rerun()


def main():
    """Main application."""
    st.set_page_config(
        page_title="AmbedkarGPT",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css()
    initialize_session_state()
    render_header()
    
    # Sidebar
    search_type, top_k = render_sidebar()
    
    # Load query engine
    if not load_query_engine():
        st.warning("Please run the indexer first to process the document.")
        st.code("python -m src.pipeline.indexer", language="bash")
        return
    
    # Main tabs
    tab1, tab2 = st.tabs(["💬 Chat", "🕸️ Knowledge Graph"])
    
    with tab1:
        render_chat_tab(search_type, top_k)
    
    with tab2:
        render_knowledge_graph_tab()


if __name__ == "__main__":
    main()
