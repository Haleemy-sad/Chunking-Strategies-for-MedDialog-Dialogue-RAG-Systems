"""
Conversational Chunking Baseline (CCB)
Rule-based dialogue-aware chunking baseline for medical conversations.
"""

from typing import List, Dict, Any
import time
import tiktoken


class ConversationalChunkingBaseline:
    """Conversation-aware baseline that preserves turn structure and QA adjacency."""

    def __init__(self, max_turns_per_chunk: int = 4):
        """
        Initialize CCB chunker.

        Args:
            max_turns_per_chunk: Maximum turns allowed in one chunk.
        """
        self.max_turns_per_chunk = max_turns_per_chunk
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.stats = {
            'chunk_count': 0,
            'total_turns': 0,
            'avg_turns_per_chunk': 0,
            'avg_chunk_length': 0,
            'processing_time': 0,
            'qa_pairs_preserved_by_rule': 0
        }

    def _normalize_utterances(self, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize utterances and attach explicit turn indices."""
        normalized = []
        for idx, utt in enumerate(dialogue.get('utterances', [])):
            text = str(utt.get('text', '')).strip()
            if not text:
                continue
            normalized.append({
                'turn_index': idx,
                'speaker': str(utt.get('speaker', 'unknown')).lower(),
                'text': text
            })
        return normalized

    def _build_chunk(self, dialogue: Dict[str, Any], turns: List[Dict[str, Any]], chunk_index: int) -> Dict[str, Any]:
        """Build one chunk dictionary in the shared project format."""
        text_parts = [f"{turn['speaker'].capitalize()}: {turn['text']}" for turn in turns]
        chunk_text = "\n".join(text_parts)
        token_count = len(self.encoding.encode(chunk_text))

        dialogue_id = dialogue.get('dialogue_id', dialogue.get('id', 'unknown'))
        turn_indices = [turn['turn_index'] for turn in turns]

        return {
            'chunk_id': f"{dialogue_id}_ccb_{chunk_index}",
            'text': chunk_text,
            'token_count': token_count,
            'turn_count': len(turns),
            'turn_indices': turn_indices,
            'turn_range': (min(turn_indices), max(turn_indices)) if turn_indices else (0, 0),
            'speakers': [turn['speaker'].capitalize() for turn in turns],
            'dialogue_id': dialogue_id,
            'chunk_index': chunk_index,
            'specialty': dialogue.get('specialty', 'unknown'),
            'strategy': 'ccb'
        }

    def chunk_dialogue(self, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk one dialogue with conversation-aware rules.

        Rules:
        1) Start from a patient initiative when possible.
        2) Keep patient question and immediate doctor response together.
        3) Group short contiguous conversation segments without splitting QA pairs.
        """
        start_time = time.time()
        turns = self._normalize_utterances(dialogue)
        if not turns:
            return []

        chunks = []
        i = 0

        while i < len(turns):
            current = []
            start_speaker = turns[i]['speaker']

            if start_speaker == 'patient':
                # Patient initiative: keep initial patient side together.
                while i < len(turns) and turns[i]['speaker'] == 'patient' and len(current) < self.max_turns_per_chunk:
                    current.append(turns[i])
                    i += 1

                # Preserve QA adjacency by forcing at least one doctor reply if available.
                doctor_added = 0
                while i < len(turns) and turns[i]['speaker'] == 'doctor':
                    if len(current) >= self.max_turns_per_chunk and doctor_added > 0:
                        break
                    current.append(turns[i])
                    doctor_added += 1
                    i += 1

                if doctor_added > 0:
                    self.stats['qa_pairs_preserved_by_rule'] += 1
            else:
                # Doctor-leading segment fallback.
                while i < len(turns) and turns[i]['speaker'] == 'doctor' and len(current) < self.max_turns_per_chunk:
                    current.append(turns[i])
                    i += 1

            if not current:
                current.append(turns[i])
                i += 1

            chunks.append(self._build_chunk(dialogue, current, len(chunks)))

        self.stats['chunk_count'] += len(chunks)
        self.stats['total_turns'] += len(turns)
        self.stats['processing_time'] += time.time() - start_time

        if self.stats['chunk_count'] > 0:
            self.stats['avg_turns_per_chunk'] = self.stats['total_turns'] / self.stats['chunk_count']
            self.stats['avg_chunk_length'] = sum(c['token_count'] for c in chunks) / len(chunks)

        return chunks

    def chunk_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple dialogues."""
        all_chunks = []
        for dialogue in dialogues:
            all_chunks.extend(self.chunk_dialogue(dialogue))
        return all_chunks

    def get_stats(self) -> Dict[str, Any]:
        """Return chunking statistics."""
        return self.stats.copy()
