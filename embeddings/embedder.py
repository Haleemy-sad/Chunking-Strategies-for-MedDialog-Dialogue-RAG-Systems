"""
Embedding Module
Handles text embedding using sentence transformers
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
import time


class TextEmbedder:
    """Embed text using sentence transformers"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = None,
                 batch_size: int = 32):
        """
        Initialize the embedder
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cuda' or 'cpu')
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
        self.stats = {
            'total_embeddings': 0,
            'total_time': 0,
            'avg_time_per_embedding': 0
        }
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        start_time = time.time()
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        elapsed = time.time() - start_time
        self.stats['total_embeddings'] += 1
        self.stats['total_time'] += elapsed
        self.stats['avg_time_per_embedding'] = self.stats['total_time'] / self.stats['total_embeddings']
        
        return embedding
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Embed multiple texts
        
        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings (shape: [num_texts, embedding_dim])
        """
        start_time = time.time()
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        
        elapsed = time.time() - start_time
        self.stats['total_embeddings'] += len(texts)
        self.stats['total_time'] += elapsed
        
        # Avoid division by zero
        if self.stats['total_embeddings'] > 0:
            self.stats['avg_time_per_embedding'] = self.stats['total_time'] / self.stats['total_embeddings']
        
        return embeddings
    
    def embed_chunks(self, chunks: List[dict], show_progress: bool = True) -> np.ndarray:
        """
        Embed chunk dictionaries (extracts 'text' field)
        
        Args:
            chunks: List of chunk dictionaries
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        texts = [chunk['text'] for chunk in chunks]
        return self.embed_texts(texts, show_progress=show_progress)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score (0 to 1)
        """
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Similarity matrix
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        
        # Compute dot product
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix
    
    def get_stats(self) -> dict:
        """Get embedding statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_embeddings': 0,
            'total_time': 0,
            'avg_time_per_embedding': 0
        }
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'batch_size': self.batch_size
        }


if __name__ == "__main__":
    # Test the embedder
    embedder = TextEmbedder()
    
    # Test single text
    text = "Patient has chest pain and shortness of breath."
    embedding = embedder.embed_text(text)
    print(f"Single embedding shape: {embedding.shape}")
    
    # Test multiple texts
    texts = [
        "Patient has chest pain.",
        "Doctor recommends ECG test.",
        "Patient has skin rash."
    ]
    embeddings = embedder.embed_texts(texts, show_progress=False)
    print(f"Multiple embeddings shape: {embeddings.shape}")
    
    # Test similarity
    sim = embedder.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between text 0 and 1: {sim:.3f}")
    
    print(f"\nStats: {embedder.get_stats()}")
    print(f"Model info: {embedder.get_model_info()}")
