"""
Retriever Module
Handles retrieval of relevant chunks from vector store
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from embeddings.embedder import TextEmbedder
from vector_store.faiss_index import FAISSIndex


class Retriever:
    """Top-K retrieval from vector store"""
    
    def __init__(self, 
                 embedder: TextEmbedder,
                 index: FAISSIndex,
                 default_top_k: int = 5):
        """
        Initialize retriever
        
        Args:
            embedder: Text embedder instance
            index: FAISS index instance
            default_top_k: Default number of results to retrieve
        """
        self.embedder = embedder
        self.index = index
        self.default_top_k = default_top_k
        
        self.stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0,
            'avg_chunks_retrieved': 0
        }
    
    def retrieve(self, 
                query: str,
                top_k: int = None) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query text
            top_k: Number of results to retrieve (None = use default)
            
        Returns:
            Tuple of (retrieved chunks, similarity scores)
        """
        if top_k is None:
            top_k = self.default_top_k
        
        import time
        start_time = time.time()
        
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Search index
        retrieved_chunks, scores = self.index.search(query_embedding, top_k=top_k)
        
        # Update stats
        retrieval_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['avg_retrieval_time'] = (
            (self.stats['avg_retrieval_time'] * (self.stats['total_queries'] - 1) + retrieval_time)
            / self.stats['total_queries']
        )
        self.stats['avg_chunks_retrieved'] = (
            (self.stats['avg_chunks_retrieved'] * (self.stats['total_queries'] - 1) + len(retrieved_chunks))
            / self.stats['total_queries']
        )
        
        return retrieved_chunks, scores
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Context {i}]")
            context_parts.append(chunk['text'])
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0,
            'avg_chunks_retrieved': 0
        }


class MultiStrategyRetriever:
    """Retriever that can switch between multiple chunking strategies"""
    
    def __init__(self, embedder: TextEmbedder):
        """
        Initialize multi-strategy retriever
        
        Args:
            embedder: Text embedder instance
        """
        self.embedder = embedder
        self.retrievers = {}
        self.current_strategy = None
    
    def add_strategy(self, strategy_name: str, index: FAISSIndex, default_top_k: int = 5):
        """
        Add a retrieval strategy
        
        Args:
            strategy_name: Name of the chunking strategy
            index: FAISS index for this strategy
            default_top_k: Default top-K for this strategy
        """
        retriever = Retriever(self.embedder, index, default_top_k)
        self.retrievers[strategy_name] = retriever
        
        if self.current_strategy is None:
            self.current_strategy = strategy_name
        
        print(f"Added retrieval strategy: {strategy_name}")
    
    def set_strategy(self, strategy_name: str):
        """
        Set the current chunking strategy
        
        Args:
            strategy_name: Name of the strategy to use
        """
        if strategy_name not in self.retrievers:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        self.current_strategy = strategy_name
        print(f"Switched to strategy: {strategy_name}")
    
    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve using current strategy
        
        Args:
            query: Query text
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (retrieved chunks, similarity scores)
        """
        if self.current_strategy is None:
            raise ValueError("No strategy set")
        
        return self.retrievers[self.current_strategy].retrieve(query, top_k)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        return list(self.retrievers.keys())
    
    def get_stats(self, strategy_name: str = None) -> Dict[str, Any]:
        """
        Get retrieval statistics for a strategy
        
        Args:
            strategy_name: Strategy to get stats for (None = current)
            
        Returns:
            Statistics dictionary
        """
        if strategy_name is None:
            strategy_name = self.current_strategy
        
        if strategy_name not in self.retrievers:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return self.retrievers[strategy_name].get_stats()


if __name__ == "__main__":
    # Test retriever
    from embeddings.embedder import TextEmbedder
    from vector_store.faiss_index import FAISSIndex
    
    # Create embedder
    embedder = TextEmbedder()
    
    # Create dummy chunks
    chunks = [
        {'chunk_id': 'chunk_0', 'text': 'Patient has chest pain and shortness of breath.'},
        {'chunk_id': 'chunk_1', 'text': 'Doctor recommends ECG and blood tests.'},
        {'chunk_id': 'chunk_2', 'text': 'Patient has skin rash on arms.'},
    ]
    
    # Embed chunks
    embeddings = embedder.embed_chunks(chunks, show_progress=False)
    
    # Build index
    index = FAISSIndex(embedder.embedding_dim, index_type="flat", metric="cosine")
    index.build_index(embeddings, chunks)
    
    # Create retriever
    retriever = Retriever(embedder, index, default_top_k=2)
    
    # Test retrieval
    query = "What are the cardiac symptoms?"
    results, scores = retriever.retrieve(query)
    
    print(f"Query: {query}\n")
    print(f"Retrieved {len(results)} chunks:")
    for i, (chunk, score) in enumerate(zip(results, scores), 1):
        print(f"  {i}. Score: {score:.3f}")
        print(f"     {chunk['text']}")
    
    print(f"\nStats: {retriever.get_stats()}")
