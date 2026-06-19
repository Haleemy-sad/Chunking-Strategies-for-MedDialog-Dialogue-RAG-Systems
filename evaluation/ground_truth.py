"""
Ground Truth Generation for Medical Dialogue Retrieval Evaluation

Creates query-document pairs with relevance judgments from dialogue structure:
- Patient questions as queries
- Doctor responses as relevant documents
- Enables proper evaluation of retrieval metrics
"""

import re
import random
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import json


class GroundTruthGenerator:
    """Generate ground truth query-document pairs from medical dialogues"""
    
    def __init__(self, random_seed=42):
        """
        Initialize ground truth generator
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        self.query_map = {}  # query -> relevant chunk IDs
        self.dialogue_map = {}  # chunk_id -> dialogue_id
        
    def create_ground_truth_from_dialogues(self,
                                          dialogues: List[Dict],
                                          chunks: List[Dict],
                                          strategy_name: str = "") -> Dict:
        """
        Create ground truth mappings from dialogue structure
        
        Args:
            dialogues: List of dialogue dictionaries
            chunks: List of chunk dictionaries with metadata
            strategy_name: Name of chunking strategy (for tracking)
            
        Returns:
            Dictionary with ground truth mappings and metadata
        """
        print(f"\n  Creating ground truth for: {strategy_name}")
        
        # Build chunk to dialogue mapping
        chunk_to_dialogue = self._map_chunks_to_dialogues(chunks, dialogues)
        
        # Extract Q&A pairs from dialogues
        qa_pairs = self._extract_qa_pairs(dialogues)
        print(f"    Extracted {len(qa_pairs)} Q&A pairs")
        
        # Map queries to relevant chunks
        query_to_chunks = self._map_queries_to_chunks(qa_pairs, chunks, chunk_to_dialogue)
        print(f"    Created {len(query_to_chunks)} query mappings")
        
        # Create test set (random sample for efficiency)
        test_queries = self._create_test_set(query_to_chunks, max_queries=500)
        print(f"    Test set: {len(test_queries)} queries")
        
        ground_truth = {
            'strategy_name': strategy_name,
            'total_dialogues': len(dialogues),
            'total_chunks': len(chunks),
            'total_qa_pairs': len(qa_pairs),
            'test_queries': test_queries,
            'query_to_relevant_chunks': query_to_chunks,
            'query_to_reference_answers': self._map_queries_to_reference_answers(qa_pairs),
            'chunk_to_dialogue': chunk_to_dialogue
        }
        
        return ground_truth
    
    def _map_chunks_to_dialogues(self, chunks: List[Dict], dialogues: List[Dict]) -> Dict:
        """Map each chunk to its source dialogue"""
        chunk_to_dialogue = {}
        
        for chunk in chunks:
            # Extract dialogue ID from chunk - try multiple locations
            chunk_id = chunk.get('id', chunk.get('chunk_id', ''))
            
            # Try metadata first, then direct field
            if 'metadata' in chunk and 'dialogue_id' in chunk['metadata']:
                dialogue_id = chunk['metadata']['dialogue_id']
            elif 'dialogue_id' in chunk:
                dialogue_id = chunk['dialogue_id']
            else:
                continue
            
            chunk_to_dialogue[chunk_id] = dialogue_id
        
        return chunk_to_dialogue
    
    def _extract_qa_pairs(self, dialogues: List[Dict]) -> List[Dict]:
        """
        Extract question-answer pairs from dialogues
        
        Each dialogue is a conversation, we extract:
        - Patient questions (queries)
        - Doctor responses (relevant answers)
        """
        qa_pairs = []
        
        for dialogue_idx, dialogue in enumerate(dialogues):
            # Get dialogue ID - try multiple field names
            dialogue_id = dialogue.get('dialogue_id', dialogue.get('id', f"dialogue_{dialogue_idx}"))
            
            # Get utterances
            if 'utterances' in dialogue:
                utterances = dialogue['utterances']
            elif 'dialogue' in dialogue:
                # Parse dialogue string format
                utterances = self._parse_dialogue_string(dialogue['dialogue'])
            else:
                continue
            
            # Extract Q&A pairs (Patient question -> Doctor response)
            current_question = None
            
            for utt in utterances:
                speaker = utt.get('speaker', '').lower()
                text = utt.get('text', '').strip()
                
                if not text:
                    continue
                
                if 'patient' in speaker:
                    current_question = text
                elif 'doctor' in speaker and current_question:
                    qa_pairs.append({
                        'dialogue_id': dialogue_id,
                        'question': current_question,
                        'answer': text,
                        'full_dialogue': dialogue
                    })
                    # Keep question for multi-turn context
                    # current_question = None  # Uncomment for single-turn only
        
        return qa_pairs
    
    def _parse_dialogue_string(self, dialogue_text: str) -> List[Dict]:
        """Parse dialogue from string format"""
        utterances = []
        
        # Common patterns: "Patient: ...", "Doctor: ...", "Dr: ...", "Pt: ..."
        pattern = r'(Patient|Doctor|Dr|Pt):\s*(.+?)(?=(?:Patient|Doctor|Dr|Pt):|$)'
        matches = re.findall(pattern, dialogue_text, re.IGNORECASE | re.DOTALL)
        
        for speaker, text in matches:
            speaker_normalized = 'patient' if speaker.lower() in ['patient', 'pt'] else 'doctor'
            utterances.append({
                'speaker': speaker_normalized,
                'text': text.strip()
            })
        
        return utterances
    
    def _map_queries_to_chunks(self,
                               qa_pairs: List[Dict],
                               chunks: List[Dict],
                               chunk_to_dialogue: Dict) -> Dict:
        """
        Map each query (question) to relevant chunk IDs
        
        Relevance criteria:
        1. Chunk from same dialogue (highly relevant)
        2. Chunk contains the answer text (most relevant)
        3. Chunk is contextually near the Q&A exchange
        """
        query_to_chunks = defaultdict(set)
        
        for qa in qa_pairs:
            query = qa['question']
            dialogue_id = qa['dialogue_id']
            answer_text = qa['answer'].lower()
            
            # Find chunks from the same dialogue
            relevant_chunks = []
            
            for chunk in chunks:
                chunk_id = chunk.get('id', chunk.get('chunk_id', ''))
                
                # Check if chunk is from same dialogue
                if chunk_to_dialogue.get(chunk_id) == dialogue_id:
                    chunk_text = chunk.get('text', '').lower()
                    
                    # Calculate relevance score
                    relevance_score = 0
                    
                    # Exact answer match (highest relevance)
                    if answer_text in chunk_text:
                        relevance_score = 3
                    # Partial overlap
                    elif any(word in chunk_text for word in answer_text.split() if len(word) > 4):
                        relevance_score = 2
                    # Same dialogue (contextually relevant)
                    else:
                        relevance_score = 1
                    
                    if relevance_score > 0:
                        relevant_chunks.append((chunk_id, relevance_score))
            
            # Sort by relevance and keep top chunks
            relevant_chunks.sort(key=lambda x: x[1], reverse=True)
            top_chunks = [chunk_id for chunk_id, score in relevant_chunks[:5]]
            
            if top_chunks:
                query_to_chunks[query] = set(top_chunks)
        
        return dict(query_to_chunks)
    
    def _create_test_set(self, query_to_chunks: Dict, max_queries: int = 500) -> List[str]:
        """Create test query set (random sample)"""
        all_queries = list(query_to_chunks.keys())
        
        # Filter queries with at least one relevant chunk
        valid_queries = [q for q in all_queries if len(query_to_chunks[q]) > 0]
        
        # Sample
        if len(valid_queries) > max_queries:
            test_queries = random.sample(valid_queries, max_queries)
        else:
            test_queries = valid_queries
        
        return test_queries

    def _map_queries_to_reference_answers(self, qa_pairs: List[Dict]) -> Dict[str, List[str]]:
        """Map each query to one or more reference answers."""
        query_to_answers = defaultdict(list)

        for qa in qa_pairs:
            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()

            if question and answer and answer not in query_to_answers[question]:
                query_to_answers[question].append(answer)

        return dict(query_to_answers)
    
    def get_relevant_chunks(self, query: str) -> Set[str]:
        """Get relevant chunk IDs for a query"""
        return self.query_map.get(query, set())
    
    def save_ground_truth(self, ground_truth: Dict, filepath: str):
        """Save ground truth to file"""
        # Convert sets to lists for JSON serialization
        serializable = ground_truth.copy()
        serializable['query_to_relevant_chunks'] = {
            k: list(v) for k, v in ground_truth['query_to_relevant_chunks'].items()
        }
        serializable['query_to_reference_answers'] = ground_truth.get('query_to_reference_answers', {})
        
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"    Saved ground truth to {filepath}")
    
    def load_ground_truth(self, filepath: str) -> Dict:
        """Load ground truth from file"""
        with open(filepath, 'r') as f:
            ground_truth = json.load(f)
        
        # Convert lists back to sets
        ground_truth['query_to_relevant_chunks'] = {
            k: set(v) for k, v in ground_truth['query_to_relevant_chunks'].items()
        }
        ground_truth['query_to_reference_answers'] = ground_truth.get('query_to_reference_answers', {})
        
        return ground_truth
    
    def get_statistics(self, ground_truth: Dict) -> Dict:
        """Get statistics about ground truth"""
        query_to_chunks = ground_truth['query_to_relevant_chunks']
        
        relevant_counts = [len(chunks) for chunks in query_to_chunks.values()]
        
        stats = {
            'total_queries': len(query_to_chunks),
            'total_relevant_pairs': sum(relevant_counts),
            'avg_relevant_per_query': sum(relevant_counts) / len(relevant_counts) if relevant_counts else 0,
            'min_relevant': min(relevant_counts) if relevant_counts else 0,
            'max_relevant': max(relevant_counts) if relevant_counts else 0,
            'queries_with_relevance': sum(1 for c in relevant_counts if c > 0)
        }
        
        return stats


def create_ground_truth_for_all_strategies(dialogues: List[Dict],
                                           all_strategy_chunks: Dict[str, List[Dict]]) -> Dict:
    """
    Create ground truth for all chunking strategies
    
    Args:
        dialogues: List of dialogues
        all_strategy_chunks: Dictionary mapping strategy names to their chunks
        
    Returns:
        Dictionary mapping strategy names to their ground truth
    """
    generator = GroundTruthGenerator()
    all_ground_truths = {}
    
    print("\n" + "="*80)
    print("CREATING GROUND TRUTH FOR ALL STRATEGIES")
    print("="*80)
    
    for strategy_name, chunks in all_strategy_chunks.items():
        ground_truth = generator.create_ground_truth_from_dialogues(
            dialogues=dialogues,
            chunks=chunks,
            strategy_name=strategy_name
        )
        
        # Print statistics
        stats = generator.get_statistics(ground_truth)
        print(f"    Statistics: {stats}")
        
        all_ground_truths[strategy_name] = ground_truth
    
    print("\n✓ Ground truth creation complete")
    return all_ground_truths
