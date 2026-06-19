"""
Dialogue-Turn Based Chunking Strategy
Groups utterances by conversation turns
"""

import tiktoken
from typing import List, Dict, Any
import time


class TurnBasedChunker:
    """Turn-based chunking that preserves dialogue structure"""
    
    def __init__(self, turns_per_chunk: int = 2):
        """
        Initialize turn-based chunker
        
        Args:
            turns_per_chunk: Number of dialogue turns per chunk (1, 2, or 3)
        """
        self.turns_per_chunk = turns_per_chunk
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.stats = {
            'chunk_count': 0,
            'total_turns': 0,
            'avg_turns_per_chunk': 0,
            'avg_chunk_length': 0,
            'processing_time': 0
        }
    
    def chunk_dialogue(self, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a dialogue by turns
        
        Args:
            dialogue: Dialogue with utterances
            
        Returns:
            List of chunk dictionaries
        """
        start_time = time.time()
        
        utterances = dialogue['utterances']
        chunks = []
        
        # Group utterances into chunks
        for i in range(0, len(utterances), self.turns_per_chunk):
            chunk_utterances = utterances[i:i + self.turns_per_chunk]
            
            # Build chunk text with speaker labels
            text_parts = []
            speakers = []
            for utt in chunk_utterances:
                speaker = utt['speaker'].capitalize()
                speakers.append(speaker)
                text_parts.append(f"{speaker}: {utt['text']}")
            
            chunk_text = " ".join(text_parts)
            
            # Count tokens
            token_count = len(self.encoding.encode(chunk_text))
            
            chunks.append({
                'chunk_id': f"{dialogue['dialogue_id']}_turns_{i}_{i+len(chunk_utterances)-1}",
                'text': chunk_text,
                'token_count': token_count,
                'turn_count': len(chunk_utterances),
                'turn_range': (i, i + len(chunk_utterances) - 1),
                'speakers': speakers,
                'dialogue_id': dialogue['dialogue_id'],
                'chunk_index': len(chunks),
                'specialty': dialogue.get('specialty', 'unknown'),
                'strategy': f'turn_{self.turns_per_chunk}'
            })
        
        # Update stats
        self.stats['chunk_count'] += len(chunks)
        self.stats['total_turns'] += len(utterances)
        self.stats['processing_time'] += time.time() - start_time
        
        if self.stats['chunk_count'] > 0:
            self.stats['avg_turns_per_chunk'] = self.stats['total_turns'] / self.stats['chunk_count']
            total_tokens = sum(c['token_count'] for c in chunks)
            self.stats['avg_chunk_length'] = total_tokens / self.stats['chunk_count']
        
        return chunks
    
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
            'total_turns': 0,
            'avg_turns_per_chunk': 0,
            'avg_chunk_length': 0,
            'processing_time': 0
        }


class SingleTurnChunker(TurnBasedChunker):
    """Each chunk is exactly one turn"""
    
    def __init__(self):
        super().__init__(turns_per_chunk=1)


class DoubleTurnChunker(TurnBasedChunker):
    """Each chunk is two turns (typical Q&A pair)"""
    
    def __init__(self):
        super().__init__(turns_per_chunk=2)


class TripleTurnChunker(TurnBasedChunker):
    """Each chunk is three turns (extended context)"""
    
    def __init__(self):
        super().__init__(turns_per_chunk=3)


if __name__ == "__main__":
    # Test the chunker
    test_dialogue = {
        'dialogue_id': 'test_001',
        'specialty': 'cardiology',
        'utterances': [
            {'speaker': 'patient', 'text': 'I have chest pain.'},
            {'speaker': 'doctor', 'text': 'Where is the pain located?'},
            {'speaker': 'patient', 'text': 'In the center of my chest.'},
            {'speaker': 'doctor', 'text': 'When did it start?'},
            {'speaker': 'patient', 'text': 'Two days ago.'},
            {'speaker': 'doctor', 'text': 'I recommend an ECG test.'},
        ]
    }
    
    print("Single Turn Chunking:")
    chunker1 = SingleTurnChunker()
    chunks1 = chunker1.chunk_dialogue(test_dialogue)
    print(f"Created {len(chunks1)} chunks")
    for chunk in chunks1[:2]:
        print(f"  {chunk['chunk_id']}: {chunk['speakers']} - {chunk['text'][:60]}...")
    
    print("\nDouble Turn Chunking:")
    chunker2 = DoubleTurnChunker()
    chunks2 = chunker2.chunk_dialogue(test_dialogue)
    print(f"Created {len(chunks2)} chunks")
    for chunk in chunks2[:2]:
        print(f"  {chunk['chunk_id']}: {chunk['speakers']} - {chunk['text'][:60]}...")
    
    print(f"\nStats: {chunker2.get_stats()}")
