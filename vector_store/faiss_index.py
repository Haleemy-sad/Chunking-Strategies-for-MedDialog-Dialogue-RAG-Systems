"""
FAISS Vector Store
Builds and searches vector indexes
"""

import faiss
import numpy as np
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any


class FAISSIndex:
    """FAISS vector store for similarity search"""
    
    def __init__(self, 
                 embedding_dim: int,
                 index_type: str = "flat",
                 metric: str = "cosine"):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ('flat' or 'hnsw')
            metric: Distance metric ('cosine' or 'l2')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        
        self.index = None
        self.chunks = []
        self.embeddings = None
        
        self.stats = {
            'num_vectors': 0,
            'index_size_mb': 0,
            'build_time': 0,
            'avg_search_time': 0,
            'total_searches': 0
        }
    
    def build_index(self, 
                   embeddings: np.ndarray,
                   chunks: List[Dict[str, Any]],
                   hnsw_m: int = 32,
                   hnsw_ef_construction: int = 200):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Array of embeddings (shape: [num_vectors, embedding_dim])
            chunks: List of chunk dictionaries corresponding to embeddings
            hnsw_m: HNSW parameter M (number of connections)
            hnsw_ef_construction: HNSW parameter ef_construction
        """
        start_time = time.time()
        
        print(f"Building {self.index_type} index with {len(embeddings)} vectors...")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Create index based on type
        if self.index_type == "flat":
            # Flat index (exhaustive search)
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product after normalization = cosine
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        elif self.index_type == "hnsw":
            # HNSW index (approximate search)
            if self.metric == "cosine":
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, hnsw_m, faiss.METRIC_L2)
            
            self.index.hnsw.efConstruction = hnsw_ef_construction
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add vectors to index
        self.index.add(embeddings)
        
        # Store chunks and embeddings
        self.chunks = chunks
        self.embeddings = embeddings
        
        # Update stats
        build_time = time.time() - start_time
        self.stats['num_vectors'] = len(embeddings)
        self.stats['build_time'] = build_time
        
        # Estimate index size (approximate)
        if self.index_type == "flat":
            size_bytes = len(embeddings) * self.embedding_dim * 4  # 4 bytes per float32
        else:
            size_bytes = len(embeddings) * self.embedding_dim * 4 * 2  # Rough estimate for HNSW
        
        self.stats['index_size_mb'] = size_bytes / (1024 * 1024)
        
        print(f"Index built in {build_time:.2f}s")
        print(f"Index size: {self.stats['index_size_mb']:.2f} MB")
    
    def search(self, 
              query_embedding: np.ndarray,
              top_k: int = 5,
              ef_search: int = 64) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            ef_search: HNSW parameter ef_search (for HNSW index)
            
        Returns:
            Tuple of (retrieved chunks, similarity scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        start_time = time.time()
        
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # Set ef_search for HNSW
        if self.index_type == "hnsw":
            self.index.hnsw.efSearch = ef_search
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get chunks and scores
        retrieved_chunks = []
        scores = []
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['search_score'] = float(dist)
                retrieved_chunks.append(chunk)
                scores.append(float(dist))
        
        # Update stats
        search_time = time.time() - start_time
        self.stats['total_searches'] += 1
        self.stats['avg_search_time'] = (
            (self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + search_time) 
            / self.stats['total_searches']
        )
        
        return retrieved_chunks, scores
    
    def save_index(self, save_dir: str, strategy_name: str):
        """
        Save index and chunks to disk
        
        Args:
            save_dir: Directory to save index
            strategy_name: Name of chunking strategy
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = save_path / f"faiss_index_{strategy_name}.bin"
        faiss.write_index(self.index, str(index_file))
        
        # Save chunks
        chunks_file = save_path / f"chunks_{strategy_name}.pkl"
        with open(chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save embeddings
        embeddings_file = save_path / f"embeddings_{strategy_name}.npy"
        np.save(embeddings_file, self.embeddings)
        
        # Save stats
        stats_file = save_path / f"stats_{strategy_name}.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(self.stats, f)
        
        print(f"Index saved to {save_path}")
    
    def load_index(self, save_dir: str, strategy_name: str):
        """
        Load index and chunks from disk
        
        Args:
            save_dir: Directory containing saved index
            strategy_name: Name of chunking strategy
        """
        save_path = Path(save_dir)
        
        # Load FAISS index
        index_file = save_path / f"faiss_index_{strategy_name}.bin"
        self.index = faiss.read_index(str(index_file))
        
        # Load chunks
        chunks_file = save_path / f"chunks_{strategy_name}.pkl"
        with open(chunks_file, 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Load embeddings
        embeddings_file = save_path / f"embeddings_{strategy_name}.npy"
        self.embeddings = np.load(embeddings_file)
        
        # Load stats
        stats_file = save_path / f"stats_{strategy_name}.pkl"
        with open(stats_file, 'rb') as f:
            self.stats = pickle.load(f)
        
        print(f"Index loaded from {save_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return self.stats.copy()
    
    def get_index_size_mb(self) -> float:
        """Get index size in MB"""
        return self.stats.get('index_size_mb', 0.0)


if __name__ == "__main__":
    # Test FAISS index
    embedding_dim = 384
    num_vectors = 100
    
    # Create random embeddings for testing
    embeddings = np.random.rand(num_vectors, embedding_dim).astype(np.float32)
    
    # Create dummy chunks
    chunks = [
        {'chunk_id': f'chunk_{i}', 'text': f'This is chunk {i}', 'token_count': 10}
        for i in range(num_vectors)
    ]
    
    # Build flat index
    print("Testing Flat Index:")
    flat_index = FAISSIndex(embedding_dim, index_type="flat", metric="cosine")
    flat_index.build_index(embeddings, chunks)
    
    # Search
    query = np.random.rand(embedding_dim).astype(np.float32)
    results, scores = flat_index.search(query, top_k=5)
    
    print(f"\nTop 5 results:")
    for i, (chunk, score) in enumerate(zip(results, scores), 1):
        print(f"  {i}. {chunk['chunk_id']} (score: {score:.3f})")
    
    print(f"\nStats: {flat_index.get_stats()}")
