"""
Semantic Chunking Strategy
Groups utterances by semantic similarity
"""

import tiktoken
from typing import List, Dict, Any
import time
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticChunker:
    """Semantic chunking using sentence embeddings"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.75,
                 use_medical_entities: bool = False,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize semantic chunker
        
        Args:
            similarity_threshold: Threshold for grouping similar utterances
            use_medical_entities: Whether to use medical entity detection (requires scispaCy)
            embedding_model_name: Name of the sentence transformer model
        """
        self.similarity_threshold = similarity_threshold
        self.use_medical_entities = use_medical_entities
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Load sentence transformer
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)
        
        # Load medical entity detector if needed
        self.nlp = None
        if use_medical_entities:
            try:
                import spacy
                self.nlp = spacy.load("en_core_sci_sm")
                print("Loaded scispaCy model for medical entity detection")
            except:
                print("Warning: scispaCy not available, medical entity detection disabled")
                self.use_medical_entities = False
        
        self.stats = {
            'chunk_count': 0,
            'total_utterances': 0,
            'avg_similarity': 0,
            'avg_chunk_length': 0,
            'processing_time': 0
        }
    
    def extract_medical_entities(self, text: str) -> List[str]:
        """
        Extract medical entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of medical entities
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0 to 1)
        """
        # Get embeddings
        emb1 = self.embedder.encode(text1, convert_to_tensor=False)
        emb2 = self.embedder.encode(text2, convert_to_tensor=False)
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Boost similarity if medical entities match
        if self.use_medical_entities:
            entities1 = set(self.extract_medical_entities(text1))
            entities2 = set(self.extract_medical_entities(text2))
            
            if entities1 and entities2:
                entity_overlap = len(entities1 & entities2) / max(len(entities1), len(entities2))
                similarity = 0.7 * similarity + 0.3 * entity_overlap
        
        return float(similarity)
    
    def chunk_dialogue(self, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a dialogue semantically using batch embeddings (optimized)
        
        Args:
            dialogue: Dialogue with utterances
            
        Returns:
            List of chunk dictionaries
        """
        start_time = time.time()
        
        utterances = [
            {
                **utt,
                'turn_index': idx
            }
            for idx, utt in enumerate(dialogue['utterances'])
        ]
        
        if len(utterances) == 0:
            return []
        
        # **OPTIMIZATION: Batch encode all utterances ONCE**
        utterance_texts = [utt['text'] for utt in utterances]
        embeddings = self.embedder.encode(utterance_texts, convert_to_tensor=False)
        
        # Initialize first chunk
        chunks = []
        current_chunk_idx = 0
        current_chunk_utterances = [utterances[0]]
        
        similarities = []
        
        # Group utterances by similarity (using pre-computed embeddings)
        for i in range(1, len(utterances)):
            utt = utterances[i]
            
            # Compute similarity with current chunk's first utterance (representative)
            emb_current = embeddings[current_chunk_idx]
            emb_utt = embeddings[i]
            
            # Cosine similarity
            similarity = np.dot(emb_current, emb_utt) / (np.linalg.norm(emb_current) * np.linalg.norm(emb_utt))
            similarities.append(similarity)
            
            if similarity >= self.similarity_threshold:
                # Add to current chunk
                current_chunk_utterances.append(utt)
            else:
                # Finalize current chunk and start new one
                chunk = self._create_chunk(
                    current_chunk_utterances, 
                    dialogue['dialogue_id'],
                    len(chunks),
                    dialogue.get('specialty', 'unknown')
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_idx = i
                current_chunk_utterances = [utt]
        
        # Add final chunk
        if current_chunk_utterances:
            chunk = self._create_chunk(
                current_chunk_utterances,
                dialogue['dialogue_id'],
                len(chunks),
                dialogue.get('specialty', 'unknown')
            )
            chunks.append(chunk)
        
        # Update stats
        self.stats['chunk_count'] += len(chunks)
        self.stats['total_utterances'] += len(utterances)
        if similarities:
            self.stats['avg_similarity'] = np.mean(similarities)
        self.stats['processing_time'] += time.time() - start_time
        
        if self.stats['chunk_count'] > 0:
            total_tokens = sum(c['token_count'] for c in chunks)
            self.stats['avg_chunk_length'] = total_tokens / self.stats['chunk_count']
        
        return chunks
    
    def _create_chunk(self, utterances: List[Dict[str, str]], 
                     dialogue_id: str, chunk_index: int, specialty: str) -> Dict[str, Any]:
        """Create a chunk from utterances"""
        text_parts = []
        speakers = []
        
        for utt in utterances:
            speaker = utt['speaker'].capitalize()
            speakers.append(speaker)
            text_parts.append(f"{speaker}: {utt['text']}")
        
        chunk_text = " ".join(text_parts)
        token_count = len(self.encoding.encode(chunk_text))
        
        # Extract medical entities if enabled
        medical_entities = []
        if self.use_medical_entities:
            medical_entities = self.extract_medical_entities(chunk_text)
        
        return {
            'chunk_id': f"{dialogue_id}_semantic_{chunk_index}",
            'text': chunk_text,
            'token_count': token_count,
            'utterance_count': len(utterances),
            'turn_indices': [utt.get('turn_index') for utt in utterances if 'turn_index' in utt],
            'speakers': speakers,
            'dialogue_id': dialogue_id,
            'chunk_index': chunk_index,
            'specialty': specialty,
            'medical_entities': medical_entities,
            'strategy': f'semantic_{int(self.similarity_threshold*100)}'
        }
    
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
            'total_utterances': 0,
            'avg_similarity': 0,
            'avg_chunk_length': 0,
            'processing_time': 0
        }


if __name__ == "__main__":
    # Test the chunker
    test_dialogue = {
        'dialogue_id': 'test_001',
        'specialty': 'cardiology',
        'utterances': [
            {'speaker': 'patient', 'text': 'I have chest pain and shortness of breath.'},
            {'speaker': 'doctor', 'text': 'These are concerning cardiac symptoms.'},
            {'speaker': 'patient', 'text': 'Also, I have a skin rash on my arm.'},
            {'speaker': 'doctor', 'text': 'The rash is likely unrelated to your chest symptoms.'},
        ]
    }
    
    print("Semantic Chunking:")
    chunker = SemanticChunker(similarity_threshold=0.75, use_medical_entities=False)
    chunks = chunker.chunk_dialogue(test_dialogue)
    
    print(f"Created {len(chunks)} semantic chunks")
    for chunk in chunks:
        print(f"\n{chunk['chunk_id']}:")
        print(f"  Utterances: {chunk['utterance_count']}")
        print(f"  Tokens: {chunk['token_count']}")
        print(f"  Text: {chunk['text'][:100]}...")
    
    print(f"\nStats: {chunker.get_stats()}")
