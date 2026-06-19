"""
Streamlit UI for Medical RAG System
Interactive interface for comparing chunking strategies
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.meddialog_loader import MedDialogLoader
from preprocessing.cleaner import TextCleaner
from chunking.fixed import FixedLengthChunker
from chunking.overlap import OverlappingWindowChunker
from chunking.turn_based import SingleTurnChunker, DoubleTurnChunker, TripleTurnChunker
from chunking.semantic import SemanticChunker
from embeddings.embedder import TextEmbedder
from vector_store.faiss_index import FAISSIndex
from retrieval.retriever import MultiStrategyRetriever
from evaluation.retrieval_metrics import RetrievalMetrics
from evaluation.hallucination import HallucinationEvaluator
from evaluation.efficiency import EfficiencyMetrics


# Page configuration
st.set_page_config(
    page_title="Medical RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chunk-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    """Load and preprocess dataset"""
    loader = MedDialogLoader()
    # Load 500 dialogues for testing from local MedDialog file
    file_path = str(Path(__file__).parent.parent / "data" / "downloaded" / "meddialog.json")
    dialogues = loader.load_from_local(file_path=file_path, max_samples=500)
    
    cleaner = TextCleaner()
    cleaned_dialogues = cleaner.clean_dialogues(dialogues)
    
    return cleaned_dialogues


@st.cache_resource
def initialize_embedder():
    """Initialize text embedder"""
    return TextEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=32
    )


@st.cache_resource
def build_indexes(dialogues, _embedder):
    """Build vector indexes for all chunking strategies"""
    
    strategies = {
        'Fixed-256': FixedLengthChunker(token_size=256),
        'Fixed-512': FixedLengthChunker(token_size=512),
        'Overlap-512-30%': OverlappingWindowChunker(window_size=512, overlap_ratio=0.3),
        'Turn-Single': SingleTurnChunker(),
        'Turn-Double': DoubleTurnChunker(),
        'Semantic-75%': SemanticChunker(similarity_threshold=0.75, use_medical_entities=False)
    }
    
    indexes = {}
    
    with st.spinner("Building vector indexes for all strategies..."):
        progress_bar = st.progress(0)
        
        for idx, (name, chunker) in enumerate(strategies.items()):
            st.write(f"Processing: {name}")
            
            # Chunk dialogues
            chunks = chunker.chunk_dialogues(dialogues)
            
            # Embed chunks
            embeddings = _embedder.embed_chunks(chunks, show_progress=False)
            
            # Build FAISS index
            index = FAISSIndex(_embedder.embedding_dim, index_type="flat", metric="cosine")
            index.build_index(embeddings, chunks)
            
            indexes[name] = {
                'chunker': chunker,
                'index': index,
                'chunks': chunks,
                'stats': chunker.get_stats()
            }
            
            progress_bar.progress((idx + 1) / len(strategies))
        
        progress_bar.empty()
    
    return indexes


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">🏥 Medical RAG System</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Compare Chunking Strategies for Medical Dialogue Retrieval</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Load data
    if 'dialogues' not in st.session_state:
        with st.spinner("Loading medical dialogues..."):
            st.session_state.dialogues = load_data()
            st.session_state.embedder = initialize_embedder()
    
    # Build indexes
    if 'indexes' not in st.session_state:
        st.session_state.indexes = build_indexes(
            st.session_state.dialogues,
            st.session_state.embedder
        )
        st.success(f"✅ Loaded {len(st.session_state.dialogues)} dialogues and built {len(st.session_state.indexes)} indexes")
    
    # Strategy selection
    st.sidebar.markdown("### 📊 Chunking Strategy")
    strategy_name = st.sidebar.selectbox(
        "Select Strategy",
        list(st.session_state.indexes.keys())
    )
    
    # Top-K selection
    st.sidebar.markdown("### 🔢 Retrieval Settings")
    top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=10, value=5)
    
    # Show strategy stats
    st.sidebar.markdown("### 📈 Strategy Statistics")
    strategy_info = st.session_state.indexes[strategy_name]
    stats = strategy_info['stats']
    
    st.sidebar.metric("Total Chunks", stats.get('chunk_count', 'N/A'))
    st.sidebar.metric("Avg Chunk Length", f"{stats.get('avg_chunk_length', 0):.0f} tokens")
    st.sidebar.metric("Index Size", f"{strategy_info['index'].stats['index_size_mb']:.2f} MB")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Query System", "📊 Compare Strategies", "📖 Dataset Info"])
    
    with tab1:
        st.markdown('<div class="sub-header">Query the Medical RAG System</div>', unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "Enter your medical question:",
            placeholder="e.g., What are the symptoms of chest pain?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_button and query:
            with st.spinner("Processing query..."):
                start_time = time.time()
                
                # Get retriever for selected strategy
                embedder = st.session_state.embedder
                index = strategy_info['index']
                
                # Retrieve
                query_embedding = embedder.embed_text(query)
                retrieved_chunks, scores = index.search(query_embedding, top_k=top_k)
                
                retrieval_time = time.time() - start_time
                
                # Display results
                st.markdown("---")
                st.markdown("### 📝 Retrieved Chunks")
                st.caption(f"Showing top {len(retrieved_chunks)} most relevant medical dialogue excerpts")
                
                for i, (chunk, score) in enumerate(zip(retrieved_chunks, scores), 1):
                    with st.expander(f"**Chunk {i}** - Similarity: {score:.3f} - Strategy: {chunk['strategy']}", expanded=(i<=2)):
                        st.markdown(f'<div class="chunk-box">{chunk["text"]}</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Tokens", chunk.get('token_count', 'N/A'))
                        col2.metric("Dialogue ID", chunk.get('dialogue_id', 'N/A'))
                        col3.metric("Chunk Index", chunk.get('chunk_index', 'N/A'))
                
                # Metrics
                st.markdown("---")
                st.markdown("### 📊 Query Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retrieval Time", f"{retrieval_time:.3f}s")
                col2.metric("Chunks Retrieved", len(retrieved_chunks))
                col3.metric("Top Score", f"{scores[0]:.3f}" if scores else "N/A")
                col4.metric("Strategy", strategy_name)
                # Generation has been removed from this public copy to avoid heavy model imports
                st.markdown("---")
                st.markdown("### 💬 Generation Disabled")
                st.info("Answer generation is disabled in this public copy. Showing retrieved context instead.")

                # Prepare context from retrieved chunks and display a preview
                context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
                st.markdown("#### Retrieved Context Preview")
                st.info(context[:1000] + "..." if len(context) > 1000 else context)
    
    with tab2:
        st.markdown('<div class="sub-header">Compare Chunking Strategies</div>', unsafe_allow_html=True)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, info in st.session_state.indexes.items():
            stats = info['stats']
            index_stats = info['index'].stats
            
            comparison_data.append({
                'Strategy': name,
                'Num Chunks': stats.get('chunk_count', 0),
                'Avg Chunk Length (tokens)': f"{stats.get('avg_chunk_length', 0):.0f}",
                'Index Size (MB)': f"{index_stats['index_size_mb']:.2f}",
                'Build Time (s)': f"{index_stats['build_time']:.2f}",
            })
        
        df = pd.DataFrame(comparison_data)
        
        st.dataframe(df, use_container_width=True)
        
        # Visualizations
        st.markdown("### 📈 Visual Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(df.set_index('Strategy')['Num Chunks'])
            st.caption("Number of chunks per strategy")
        
        with col2:
            # Convert to numeric for plotting
            df_plot = df.copy()
            df_plot['Index Size (MB)'] = df_plot['Index Size (MB)'].astype(float)
            st.bar_chart(df_plot.set_index('Strategy')['Index Size (MB)'])
            st.caption("Index size per strategy (MB)")
    
    with tab3:
        st.markdown('<div class="sub-header">Dataset Information</div>', unsafe_allow_html=True)
        
        # Dataset stats
        loader = MedDialogLoader()
        loader.data = st.session_state.dialogues
        stats = loader.get_statistics()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Dialogues", stats['total_dialogues'])
        col2.metric("Total Turns", stats['total_turns'])
        col3.metric("Avg Turns/Dialogue", f"{stats['avg_turns_per_dialogue']:.1f}")
        
        # Specialties distribution
        st.markdown("### 🏥 Medical Specialties")
        specialty_df = pd.DataFrame(
            list(stats['specialties'].items()),
            columns=['Specialty', 'Count']
        ).sort_values('Count', ascending=False)
        
        st.dataframe(specialty_df, use_container_width=True)
        
        # Sample dialogue
        st.markdown("### 📋 Sample Dialogue")
        sample = st.session_state.dialogues[0]
        
        st.write(f"**Dialogue ID:** {sample['dialogue_id']}")
        st.write(f"**Specialty:** {sample['specialty']}")
        st.write(f"**Turns:** {sample['turn_count']}")
        
        for i, utt in enumerate(sample['utterances'][:4], 1):
            st.markdown(f"**Turn {i} - {utt['speaker'].capitalize()}:** {utt['text']}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: gray; font-size: 0.9rem;">'
        'Medical RAG System for Research | Built with Streamlit'
        '</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
