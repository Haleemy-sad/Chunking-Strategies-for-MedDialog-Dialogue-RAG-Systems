"""
Recursive Dialogue Chunking Strategy
Recursively splits dialogue text by natural boundaries until chunk size constraints are met.
"""

import tiktoken
from typing import List, Dict, Any
import time


class RecursiveDialogueChunker:
    """Recursive chunking with separator hierarchy and token-size control."""

    def __init__(
        self,
        token_size: int = 512,
        min_chunk_tokens: int = 80,
        separators: List[str] = None
    ):
        self.token_size = token_size
        self.min_chunk_tokens = min_chunk_tokens
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
        self.encoding = tiktoken.get_encoding("cl100k_base")

        self.stats = {
            'chunk_count': 0,
            'total_tokens': 0,
            'avg_chunk_length': 0,
            'processing_time': 0,
            'recursive_splits': 0
        }

    def _token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _fallback_token_split(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)
        parts = []
        for start in range(0, len(tokens), self.token_size):
            part_tokens = tokens[start:start + self.token_size]
            parts.append(self.encoding.decode(part_tokens).strip())
        return [p for p in parts if p]

    def _recursive_split(self, text: str, sep_index: int = 0) -> List[str]:
        text = text.strip()
        if not text:
            return []

        token_count = self._token_count(text)
        if token_count <= self.token_size:
            return [text]

        if sep_index >= len(self.separators):
            self.stats['recursive_splits'] += 1
            return self._fallback_token_split(text)

        sep = self.separators[sep_index]
        pieces = text.split(sep)

        if len(pieces) == 1:
            return self._recursive_split(text, sep_index + 1)

        candidate_chunks = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            proposed = f"{current}{sep}{piece}" if current else piece
            proposed_tokens = self._token_count(proposed)

            if proposed_tokens <= self.token_size:
                current = proposed
            else:
                if current:
                    candidate_chunks.append(current.strip())
                current = piece

        if current:
            candidate_chunks.append(current.strip())

        final_chunks = []
        for chunk in candidate_chunks:
            if self._token_count(chunk) > self.token_size:
                self.stats['recursive_splits'] += 1
                final_chunks.extend(self._recursive_split(chunk, sep_index + 1))
            else:
                final_chunks.append(chunk)

        # Merge tiny adjacent chunks when possible
        merged = []
        for chunk in final_chunks:
            if not merged:
                merged.append(chunk)
                continue

            if self._token_count(chunk) < self.min_chunk_tokens:
                combined = f"{merged[-1]} {chunk}".strip()
                if self._token_count(combined) <= self.token_size:
                    merged[-1] = combined
                    continue
            merged.append(chunk)

        return merged

    def chunk_text(self, text: str, dialogue_id: str = None) -> List[Dict[str, Any]]:
        start_time = time.time()

        chunk_texts = self._recursive_split(text)
        chunks = []

        for idx, chunk_text in enumerate(chunk_texts):
            token_count = self._token_count(chunk_text)
            chunks.append({
                'chunk_id': f"{dialogue_id}_recursive_{idx}" if dialogue_id else f"recursive_{idx}",
                'text': chunk_text,
                'token_count': token_count,
                'dialogue_id': dialogue_id,
                'chunk_index': idx,
                'strategy': f"recursive_{self.token_size}"
            })

        total_tokens = self._token_count(text)
        self.stats['chunk_count'] += len(chunks)
        self.stats['total_tokens'] += total_tokens
        self.stats['processing_time'] += time.time() - start_time

        if self.stats['chunk_count'] > 0:
            self.stats['avg_chunk_length'] = self.stats['total_tokens'] / self.stats['chunk_count']

        return chunks

    def chunk_dialogue(self, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        text_parts = []
        for utt in dialogue['utterances']:
            speaker = utt['speaker'].capitalize()
            text_parts.append(f"{speaker}: {utt['text']}")

        full_text = "\n".join(text_parts)
        return self.chunk_text(full_text, dialogue.get('dialogue_id', dialogue.get('id')))

    def chunk_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        for dialogue in dialogues:
            all_chunks.extend(self.chunk_dialogue(dialogue))
        return all_chunks

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def reset_stats(self):
        self.stats = {
            'chunk_count': 0,
            'total_tokens': 0,
            'avg_chunk_length': 0,
            'processing_time': 0,
            'recursive_splits': 0
        }
