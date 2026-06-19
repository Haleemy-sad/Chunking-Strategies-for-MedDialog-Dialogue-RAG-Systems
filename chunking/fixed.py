"""
Fixed-Length Chunking Strategy
Splits dialogues into fixed token-size chunks
"""

import tiktoken
from typing import List, Dict, Any
import time


class FixedLengthChunker:
    """Fixed-length chunking without overlap"""
    
    def __init__(self, token_size: int = 512):
        """
        Initialize fixed-length chunker
        
        Args:
            token_size: Number of tokens per chunk (256, 512, or 1024)
        """
        self.token_size = token_size
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.stats = {
            'chunk_count': 0,
            'total_tokens': 0,
            'avg_chunk_length': 0,
            'processing_time': 0
        }
    
    def chunk_text(self, text: str, dialogue_id: str = None) -> List[Dict[str, Any]]:
        """
        Chunk a single text into fixed-length chunks
        
        Args:
            text: Input text
            dialogue_id: Optional dialogue ID for tracking
            
        Returns:
            List of chunk dictionaries
        """
        start_time = time.time()
        
        # Tokenize
        tokens = self.encoding.encode(text)
        
        chunks = []
        for i in range(0, len(tokens), self.token_size):
            chunk_tokens = tokens[i:i + self.token_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append({
                'chunk_id': f"{dialogue_id}_chunk_{len(chunks)}" if dialogue_id else f"chunk_{len(chunks)}",
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'dialogue_id': dialogue_id,
                'chunk_index': len(chunks),
                'strategy': f'fixed_{self.token_size}'
            })
        
        # Update stats
        self.stats['chunk_count'] += len(chunks)
        self.stats['total_tokens'] += len(tokens)
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
            'processing_time': 0
        }


if __name__ == "__main__":
    # Test the chunker
    chunker = FixedLengthChunker(token_size=256)
    
    test_text = "Patient: I have been experiencing chest pain. " * 50
    chunks = chunker.chunk_text(test_text, "test_001")
    
    print(f"Created {len(chunks)} chunks")
    for chunk in chunks[:2]:
        print(f"\n{chunk['chunk_id']}: {chunk['token_count']} tokens")
        print(f"Text preview: {chunk['text'][:100]}...")
    
    print(f"\nStats: {chunker.get_stats()}")
