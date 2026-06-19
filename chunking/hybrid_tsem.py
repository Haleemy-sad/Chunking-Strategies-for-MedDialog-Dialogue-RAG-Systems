"""
T-Sem RAG: Turn-Consistent Semantic Chunking Strategy
Hybrid approach combining turn-based structure with semantic similarity
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import tiktoken


class TSemChunker:
    """
    Turn-Consistent Semantic (T-Sem) Chunking
    
    Combines turn-based dialogue structure with semantic similarity merging.
    Merges consecutive turns if their semantic similarity exceeds threshold.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.75,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu",
                 max_turns_per_chunk: int = 2):
        """
        Initialize T-Sem chunker
        
        Args:
            similarity_threshold: Cosine similarity threshold for merging (default: 0.75)
            model_name: Sentence transformer model for embeddings
            device: Device for embedding model ('cpu' or 'cuda')
        """
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.device = device
        self.max_turns_per_chunk = max_turns_per_chunk
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name, device=device)
        
        # Initialize tokenizer for chunk size calculation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.stats = {
            'chunk_count': 0,
            'avg_chunk_length': 0,
            'avg_similarity': 0,
            'merge_count': 0,
            'total_turns': 0
        }
    
    def _extract_turns(self, dialogue: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract individual turns from dialogue
        
        Args:
            dialogue: Dialogue dictionary with utterances
            
        Returns:
            List of turn dictionaries with speaker and text
        """
        turns = []
        
        dialogue_id = dialogue.get('dialogue_id', dialogue.get('id', 'unknown'))

        for utterance in dialogue.get('utterances', []):
            speaker = utterance.get('speaker', 'Unknown').lower()
            text = utterance.get('text', '').strip()
            
            if text:
                turns.append({
                    'speaker': speaker,
                    'text': text,
                    'dialogue_id': dialogue_id,
                    'turn_index': len(turns)
                })
        
        return turns
    
    def _compute_similarities_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between consecutive embeddings (vectorized)
        
        Args:
            embeddings: Array of shape (n_turns, embedding_dim)
            
        Returns:
            Array of shape (n_turns-1,) with similarities between consecutive turns
        """
        if len(embeddings) < 2:
            return np.array([])
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / (norms + 1e-8)
        
        # Compute similarities with next turn: dot product of normalized vectors
        similarities = np.sum(embeddings_normalized[:-1] * embeddings_normalized[1:], axis=1)
        
        return similarities
    
    def _merge_turns(self, turns: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Merge consecutive turns based on semantic similarity (optimized with batch embedding)
        
        Args:
            turns: List of individual turns
            
        Returns:
            List of merged chunks
        """
        if not turns:
            return []
        
        # **OPTIMIZATION 1: Batch embed all turns once**
        turn_texts = [turn['text'] for turn in turns]
        embeddings = self.embedder.encode(turn_texts, convert_to_tensor=False)
        
        # **OPTIMIZATION 2: Vectorized similarity computation**
        similarities = self._compute_similarities_batch(embeddings)
        
        chunks = []
        current_chunk_turns = [turns[0]]
        merge_count = 0
        
        for i in range(1, len(turns)):
            prev_turn = turns[i-1]
            curr_turn = turns[i]
            
            # Get pre-computed similarity
            similarity = similarities[i-1] if i-1 < len(similarities) else 0.0
            
            # Turn-consistent merge rule:
            # 1) Always prioritize patient -> doctor adjacency to preserve QA units.
            # 2) Otherwise, allow semantic merge only if speakers alternate and similarity is high.
            qa_pair = prev_turn['speaker'] == 'patient' and curr_turn['speaker'] == 'doctor'
            semantic_merge = similarity >= self.similarity_threshold and prev_turn['speaker'] != curr_turn['speaker']
            can_merge = len(current_chunk_turns) < self.max_turns_per_chunk

            if can_merge and (qa_pair or semantic_merge):
                current_chunk_turns.append(curr_turn)
                merge_count += 1
            else:
                # Create chunk from accumulated turns
                chunk = self._create_chunk(current_chunk_turns)
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_turns = [curr_turn]
        
        # Add final chunk
        if current_chunk_turns:
            chunk = self._create_chunk(current_chunk_turns)
            chunks.append(chunk)
        
        # Update stats
        self.stats['avg_similarity'] = np.mean(similarities) if len(similarities) > 0 else 0
        self.stats['merge_count'] += merge_count
        self.stats['total_turns'] += len(turns)
        
        return chunks
    
    def _create_chunk(self, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Create a chunk from multiple turns
        
        Args:
            turns: List of turns to combine
            
        Returns:
            Chunk dictionary
        """
        # Combine turn texts with speaker labels
        chunk_text = "\n".join([
            f"{turn['speaker'].capitalize()}: {turn['text']}"
            for turn in turns
        ])
        
        # Calculate token count
        token_count = len(self.tokenizer.encode(chunk_text))
        
        return {
            'text': chunk_text,
            'dialogue_id': turns[0]['dialogue_id'],
            'turn_count': len(turns),
            'turn_indices': [turn['turn_index'] for turn in turns if 'turn_index' in turn],
            'token_count': token_count,
            'speakers': [turn['speaker'] for turn in turns],
            'strategy': 'tsem'
        }
    
    def chunk_dialogue(self, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single dialogue using T-Sem strategy
        
        Args:
            dialogue: Dialogue dictionary
            
        Returns:
            List of chunks
        """
        # Extract individual turns
        turns = self._extract_turns(dialogue)
        
        if not turns:
            return []
        
        # Merge turns based on semantic similarity
        chunks = self._merge_turns(turns)
        
        return chunks
    
    def chunk_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple dialogues
        
        Args:
            dialogues: List of dialogue dictionaries
            
        Returns:
            List of all chunks with metadata
        """
        all_chunks = []
        chunk_lengths = []
        
        print(f"\nProcessing {len(dialogues)} dialogues with T-Sem strategy...")
        print(f"Similarity threshold: {self.similarity_threshold}")
        
        for idx, dialogue in enumerate(dialogues):
            if idx % 100 == 0:
                print(f"Progress: {idx}/{len(dialogues)} dialogues processed")
            
            chunks = self.chunk_dialogue(dialogue)
            
            # Add chunk index
            for chunk_idx, chunk in enumerate(chunks):
                chunk['chunk_index'] = chunk_idx
                chunk['chunk_id'] = f"{chunk['dialogue_id']}_tsem_{chunk_idx}"
                chunk_lengths.append(chunk['token_count'])
            
            all_chunks.extend(chunks)
        
        # Update final statistics
        self.stats['chunk_count'] = len(all_chunks)
        self.stats['avg_chunk_length'] = np.mean(chunk_lengths) if chunk_lengths else 0
        
        print(f"\n✅ T-Sem Chunking Complete!")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   Avg chunk length: {self.stats['avg_chunk_length']:.1f} tokens")
        print(f"   Avg similarity: {self.stats['avg_similarity']:.3f}")
        print(f"   Merges performed: {self.stats['merge_count']}")
        
        # Safe merge rate calculation
        if self.stats['total_turns'] > 0:
            merge_rate = self.stats['merge_count']/self.stats['total_turns']*100
            print(f"   Merge rate: {merge_rate:.1f}%")
        else:
            print(f"   Merge rate: N/A (no turns processed)")
        
        return all_chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        return self.stats.copy()


if __name__ == "__main__":
    # Test the T-Sem chunker
    test_dialogue = {
        'id': 'test_001',
        'utterances': [
            {'speaker': 'Patient', 'text': 'I have been experiencing severe headaches for the past week.'},
            {'speaker': 'Doctor', 'text': 'Can you describe the headaches? Are they throbbing or constant?'},
            {'speaker': 'Patient', 'text': 'They are throbbing, mostly on the left side of my head.'},
            {'speaker': 'Doctor', 'text': 'Do you have any visual disturbances or sensitivity to light?'},
            {'speaker': 'Patient', 'text': 'Yes, bright lights make it worse.'},
            {'speaker': 'Doctor', 'text': 'Based on your symptoms, this sounds like migraines. I recommend...'}
        ]
    }
    
    chunker = TSemChunker(similarity_threshold=0.75)
    chunks = chunker.chunk_dialogue(test_dialogue)
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Turns: {chunk['turn_count']}")
        print(f"  Tokens: {chunk['token_count']}")
        print(f"  Text: {chunk['text'][:100]}...")
