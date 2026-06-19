"""
Overlapping Window Chunking Strategy
Creates chunks with overlapping content
"""

import tiktoken
from typing import List, Dict, Any
import time


class OverlappingWindowChunker:
    """Overlapping window chunking strategy"""
    
    def __init__(self, window_size: int = 512, overlap_ratio: float = 0.3):
        """
        Initialize overlapping window chunker
        
        Args:
            window_size: Size of each window in tokens
            overlap_ratio: Ratio of overlap (0.0 to 1.0)
        """
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.stride = int(window_size * (1 - overlap_ratio))
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.stats = {
            'chunk_count': 0,
            'total_tokens': 0,
            'avg_chunk_length': 0,
            'avg_overlap': 0,
            'processing_time': 0
        }
    
    def chunk_text(self, text: str, dialogue_id: str = None) -> List[Dict[str, Any]]:
        """
        Chunk a single text with overlapping windows
        
        Args:
            text: Input text
            dialogue_id: Optional dialogue ID for tracking
            
        Returns:
            List of chunk dictionaries
        """
        start_time = time.time()
        
        # Tokenize
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.window_size:
            # Text is smaller than window size, return as single chunk
            return [{
                'chunk_id': f"{dialogue_id}_chunk_0" if dialogue_id else "chunk_0",
                'text': text,
                'token_count': len(tokens),
                'dialogue_id': dialogue_id,
                'chunk_index': 0,
                'strategy': f'overlap_{self.window_size}_{int(self.overlap_ratio*100)}'
            }]
        
        chunks = []
        position = 0
        
        while position < len(tokens):
            # Get window
            end_pos = min(position + self.window_size, len(tokens))
            chunk_tokens = tokens[position:end_pos]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Calculate actual overlap with previous chunk
            actual_overlap = 0
            if len(chunks) > 0:
                overlap_size = position + self.window_size - position
                actual_overlap = min(self.window_size - self.stride, len(chunk_tokens))
            
            chunks.append({
                'chunk_id': f"{dialogue_id}_chunk_{len(chunks)}" if dialogue_id else f"chunk_{len(chunks)}",
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'overlap_tokens': actual_overlap,
                'dialogue_id': dialogue_id,
                'chunk_index': len(chunks),
                'strategy': f'overlap_{self.window_size}_{int(self.overlap_ratio*100)}'
            })
            
            # Move by stride
            position += self.stride
            
            # Break if we've covered all tokens
            if end_pos >= len(tokens):
                break
        
        # Update stats
        self.stats['chunk_count'] += len(chunks)
        self.stats['total_tokens'] += len(tokens)
        total_overlap = sum(c['overlap_tokens'] for c in chunks)
        
        if len(chunks) > 1:
            self.stats['avg_overlap'] = total_overlap / (len(chunks) - 1)
        
        self.stats['processing_time'] += time.time() - start_time
        
        if self.stats['chunk_count'] > 0:
            self.stats['avg_chunk_length'] = self.stats['total_tokens'] / self.stats['chunk_count']
        
        return chunks
    
    def chunk_dialogue(self, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a dialogue dictionary
        
        Args:
            dialogue: Dialogue with utterances
            
        Returns:
            List of chunk dictionaries
        """
        # Convert dialogue to text
        text_parts = []
        for utt in dialogue['utterances']:
            speaker = utt['speaker'].capitalize()
            text_parts.append(f"{speaker}: {utt['text']}")
        
        full_text = " ".join(text_parts)
        
        return self.chunk_text(full_text, dialogue['dialogue_id'])
    
    def chunk_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple dialogues
        
        Args:
            dialogues: List of dialogue dictionaries
            
        Returns:
            List of all chunks
        """
        all_chunks = []
        
        for dialogue in dialogues:
            chunks = self.chunk_dialogue(dialogue)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'chunk_count': 0,
            'total_tokens': 0,
            'avg_chunk_length': 0,
            'avg_overlap': 0,
            'processing_time': 0
        }


if __name__ == "__main__":
    # Test the chunker
    chunker = OverlappingWindowChunker(window_size=512, overlap_ratio=0.3)
    
    test_text = "Patient: I have chest pain. Doctor: Describe the pain. " * 50
    chunks = chunker.chunk_text(test_text, "test_001")
    
    print(f"Created {len(chunks)} chunks with {chunker.overlap_ratio*100}% overlap")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}: {chunk['token_count']} tokens, {chunk['overlap_tokens']} overlap")
        print(f"Text preview: {chunk['text'][:100]}...")
    
    print(f"\nStats: {chunker.get_stats()}")
