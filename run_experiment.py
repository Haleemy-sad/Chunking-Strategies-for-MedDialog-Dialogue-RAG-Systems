"""
Minimal experiment runner for chunking strategies.

This script loads a small set of dialogues (local full file if present, otherwise
the loader's sample data), runs configured chunking strategies, builds FAISS
indexes (CPU) and writes a CSV summary with basic stats per strategy.

Designed to be lightweight and runnable on a normal desktop. It avoids heavy
generation/model-serving code.
"""
import argparse
import time
import os
import csv
from pathlib import Path

from data.meddialog_loader import MedDialogLoader
from preprocessing.cleaner import TextCleaner
from embeddings.embedder import TextEmbedder
from vector_store.faiss_index import FAISSIndex

# Chunkers
from chunking.fixed import FixedLengthChunker
from chunking.overlap import OverlappingWindowChunker
from chunking.turn_based import SingleTurnChunker, DoubleTurnChunker

try:
    from chunking.semantic import SemanticChunker
    SEMANTIC_AVAILABLE = True
except Exception:
    SEMANTIC_AVAILABLE = False


def run(args):
    # Load data (prefer local downloaded file)
    loader = MedDialogLoader(dataset_source='local')
    local_file = Path(__file__).parent / 'data' / 'downloaded' / 'meddialog.json'

    if local_file.exists():
        print(f"Loading local MedDialog from {local_file}")
        dialogues = loader.load_from_local(str(local_file))
    else:
        print("No local file found — using loader fallback/sample data")
        dialogues = loader._create_sample_data()

    if args.max_dialogues:
        dialogues = dialogues[: args.max_dialogues]

    # Preprocess
    cleaner = TextCleaner()
    cleaned = cleaner.clean_dialogues(dialogues)

    # Embedder (CPU by default)
    embedder = TextEmbedder(model_name=args.model_name, device='cpu', batch_size=32)

    # Strategies to run
    strategies = [
        ('Fixed-256', FixedLengthChunker(token_size=256)),
        ('Fixed-512', FixedLengthChunker(token_size=512)),
        ('Overlap-512-30', OverlappingWindowChunker(window_size=512, overlap_ratio=0.3)),
        ('Turn-Single', SingleTurnChunker()),
        ('Turn-Double', DoubleTurnChunker()),
    ]

    if SEMANTIC_AVAILABLE:
        strategies.append(('Semantic-75', SemanticChunker(similarity_threshold=0.75, use_medical_entities=False)))
    else:
        print('SemanticChunker not available — skipping semantic strategy (requires sentence-transformers)')

    results = []

    os.makedirs('results', exist_ok=True)

    for name, chunker in strategies:
        print(f"\nRunning strategy: {name}")
        t0 = time.time()

        # Chunk
        chunks = chunker.chunk_dialogues(cleaned)

        # Embed (may be heavy if many chunks)
        try:
            embeddings = embedder.embed_chunks(chunks, show_progress=False)
        except Exception as e:
            print(f"Embedding failed for {name}: {e}")
            embeddings = None

        # Build FAISS index if embeddings succeeded
        index_stats = {'num_vectors': 0, 'index_size_mb': 0.0, 'build_time': 0.0}
        if embeddings is not None and len(embeddings) > 0:
            fi = FAISSIndex(embedder.embedding_dim, index_type='flat', metric='cosine')
            t1 = time.time()
            fi.build_index(embeddings, chunks)
            t2 = time.time()
            index_stats['num_vectors'] = fi.stats.get('num_vectors', 0)
            index_stats['index_size_mb'] = fi.stats.get('index_size_mb', 0.0)
            index_stats['build_time'] = fi.stats.get('build_time', t2 - t1)
        else:
            print(f"Skipping FAISS index for {name} due to missing embeddings")

        total_time = time.time() - t0

        # Record summary
        results.append({
            'strategy': name,
            'num_dialogues': len(cleaned),
            'num_chunks': len(chunks),
            'avg_chunk_tokens': sum(c.get('token_count', 0) for c in chunks) / (len(chunks) or 1),
            'index_size_mb': index_stats['index_size_mb'],
            'build_time_s': index_stats['build_time'],
            'total_time_s': total_time
        })

        print(f"Strategy {name} complete — chunks: {len(chunks)}, time: {total_time:.2f}s")

    # Write CSV summary
    csv_file = Path('results') / 'strategy_summary.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nWrote summary to {csv_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max-dialogues', type=int, default=100, help='Max dialogues to process (use small number for quick tests)')
    p.add_argument('--model-name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Sentence-transformers model name')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
