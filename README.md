# RAG Medical Dialogue System

A comprehensive Retrieval-Augmented Generation (RAG) system for medical dialogue analysis, designed for academic research comparing different chunking strategies.

## 🎯 Project Objective

Build a modular RAG pipeline with evaluation framework and UI to:
- Apply multiple chunking strategies on medical dialogue dataset (112K dialogues)
- Build separate vector indexes for each strategy
- Compare retrieval, generation, hallucination, and efficiency metrics
- Generate publication-quality visualizations and comparisons
- Interact with the system via Streamlit UI

## 🚀 Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run experiments (500 dialogues, ~15-20 min)
.\venv\Scripts\Activate.ps1
python run_experiment.py --max-dialogues 500

# 3. Generate visualizations
python visualize_results.py

# 4. View results
# → results/plots/ (7 graphs)
# → results/summary_table_for_thesis.csv
```

See [COMPLETE_WORKFLOW.md](COMPLETE_WORKFLOW.md) for full guide.

## 📊 Implemented Chunking Strategies

### 1. Fixed-Length Chunking
- Token sizes: 256, 512, 1024
- No overlap
- **Use case:** Baseline comparison

### 2. Overlapping Window Chunking
- Window size: 512 tokens
- Overlap: 30%
- **Use case:** Preserve context across boundaries

### 3. Dialogue-Turn Chunking
- Single turn (1 utterance)
- Double turn (2 utterances - Q&A pairs)
- Triple turn (3 utterances - extended context)
- **Use case:** Preserve conversational structure

### 4. Semantic Chunking
- Groups utterances by semantic similarity
- Threshold: 75%
- Optional medical entity detection (scispaCy)
- **Use case:** Semantically coherent chunks

## 🏗️ System Architecture

```
MedDialog Dataset
    ↓
Preprocessing (cleaner.py)
    ↓
Chunking Strategy (fixed/overlap/turn/semantic)
    ↓
Embedding Model (sentence-transformers)
    ↓
Vector Store (FAISS)
    ↓
Retriever (Top-K)
    ↓
LLM Generator (Mistral/LLaMA)
    ↓
Evaluation + UI
```

## 📁 Project Structure

```
rag-med-dialog/
│
├── data/
│   └── meddialog_loader.py          # Dataset loading
│
├── preprocessing/
│   └── cleaner.py                    # Text preprocessing
│
├── chunking/
│   ├── fixed.py                      # Fixed-length chunking
│   ├── overlap.py                    # Overlapping window chunking
│   ├── turn_based.py                 # Turn-based chunking
│   └── semantic.py                   # Semantic chunking
│
├── embeddings/
│   └── embedder.py                   # Sentence transformer embeddings
│
├── vector_store/
│   └── faiss_index.py                # FAISS vector indexing
│
├── retrieval/
│   └── retriever.py                  # Top-K retrieval
│
├── generation/
│   └── rag_pipeline.py               # RAG generation pipeline
│
├── evaluation/
│   ├── retrieval_metrics.py          # Recall@K, MRR, nDCG
│   ├── hallucination.py              # Hallucination detection
│   └── efficiency.py                 # Latency and resource metrics
│
├── ui/
│   └── app.py                        # Streamlit UI
│
├── config.yaml                       # Configuration file
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- CUDA-capable GPU (optional, for faster processing)

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd rag-med-dialog
```

2. **Create a virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model (optional, for semantic chunking with entities):**
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
```

## 💻 Usage

### Option 1: Streamlit UI (Recommended)

Launch the interactive UI:

```bash
streamlit run ui/app.py
```

The UI provides:
- **Query System:** Search medical dialogues with different strategies
- **Strategy Comparison:** Visual comparison of all chunking methods
- **Dataset Info:** Statistics and sample dialogues

### Option 2: Python API

Use the system programmatically:

```python
from data.meddialog_loader import MedDialogLoader
from preprocessing.cleaner import TextCleaner
from chunking.fixed import FixedLengthChunker
from embeddings.embedder import TextEmbedder
from vector_store.faiss_index import FAISSIndex
from retrieval.retriever import Retriever
from generation.rag_pipeline import MedicalRAGPipeline, RAGSystem

# Load data
loader = MedDialogLoader()
dialogues = loader.load_from_huggingface("train")

# Preprocess
cleaner = TextCleaner()
cleaned = cleaner.clean_dialogues(dialogues)

# Chunk
chunker = FixedLengthChunker(token_size=512)
chunks = chunker.chunk_dialogues(cleaned)

# Embed
embedder = TextEmbedder()
embeddings = embedder.embed_chunks(chunks)

# Build index
index = FAISSIndex(embedder.embedding_dim)
index.build_index(embeddings, chunks)

# Create retriever
retriever = Retriever(embedder, index)

# Create generator (optional)
generator = MedicalRAGPipeline()

# Create RAG system
rag = RAGSystem(retriever, generator)

# Query
result = rag.query("What are the symptoms of chest pain?", top_k=5)
print(result['answer'])
```

## 📊 Evaluation

### Retrieval Metrics

```python
from evaluation.retrieval_metrics import RetrievalMetrics

metrics = RetrievalMetrics()
results = metrics.evaluate_retrieval(
    retrieved_chunks=retrieved_chunks,
    relevant_chunk_ids={'chunk_1', 'chunk_2'},
    k_values=[3, 5, 10]
)

print(f"Recall@5: {results['recall@5']:.3f}")
print(f"MRR: {results['mrr']:.3f}")
```

### Hallucination Detection

```python
from evaluation.hallucination import HallucinationEvaluator

evaluator = HallucinationEvaluator()
result = evaluator.evaluate(
    generated_answer=answer,
    context_chunks=retrieved_chunks,
    query=query
)

print(f"Has hallucination: {result['has_hallucination']}")
print(f"Medical harm risk: {result['medical_harm_risk']}")
```

### Efficiency Metrics

```python
from evaluation.efficiency import EfficiencyMetrics

metrics = EfficiencyMetrics()

# Record metrics
metrics.record_embedding_metrics(num_chunks=100, embedding_time=2.5, embedding_dim=384)
metrics.record_retrieval_metrics(query_latency=0.05, num_retrieved=5)
metrics.record_generation_metrics(generation_time=3.2, tokens_generated=128, tokens_per_second=40)

# Get summary
metrics.print_summary()

# Save to CSV
metrics.save_to_csv("results/efficiency_metrics.csv")
```

## 🔬 Experiments

### Running Comparative Experiments

```python
# Compare all strategies
from chunking.fixed import FixedLengthChunker
from chunking.overlap import OverlappingWindowChunker
from chunking.turn_based import DoubleTurnChunker
from chunking.semantic import SemanticChunker

strategies = {
    'Fixed-512': FixedLengthChunker(512),
    'Overlap-512-30%': OverlappingWindowChunker(512, 0.3),
    'Turn-Double': DoubleTurnChunker(),
    'Semantic-75%': SemanticChunker(0.75)
}

results = {}

for name, chunker in strategies.items():
    print(f"\nEvaluating: {name}")
    
    # Chunk, embed, index, retrieve, generate, evaluate
    # ... (full pipeline)
    
    results[name] = {
        'retrieval_metrics': retrieval_results,
        'hallucination_metrics': halluc_results,
        'efficiency_metrics': efficiency_results
    }

# Save results
import pandas as pd
df = pd.DataFrame(results).T
df.to_csv("results/strategy_comparison.csv")
```

## 📝 Configuration

Edit `config.yaml` to customize:

```yaml
# Embedding Model
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"  # or "cpu"

# Generation Model
generation:
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  max_new_tokens: 256
  temperature: 0.7

# Chunking parameters
chunking:
  fixed:
    token_sizes: [256, 512, 1024]
  overlap:
    window_size: 512
    overlap_ratio: 0.3
```

## 🎓 For Academic Research

### Reproducibility

All experiments log:
- Exact chunking parameters
- Model versions
- Random seeds (where applicable)
- Timestamps
- System configuration

### Output for Thesis

The system generates CSV files for:
1. **Retrieval metrics per strategy** (`results/retrieval_comparison.csv`)
2. **Hallucination rates** (`results/hallucination_analysis.csv`)
3. **Efficiency comparison** (`results/efficiency_comparison.csv`)
4. **Per-query results** (`results/query_results.csv`)

### Citation

If using this code in research:

```
@software{medical_rag_chunking,
  title={Medical RAG System: Chunking Strategy Comparison},
  author={Your Name},
  year={2025},
  institution={Your University}
}
```

## ⚠️ Important Notes

### Medical Disclaimer

This system is for **RESEARCH PURPOSES ONLY**. It should not be used for:
- Medical diagnosis
- Treatment recommendations
- Clinical decision-making
- Patient care

Always consult qualified healthcare professionals for medical advice.

### Data Privacy

- Uses publicly available MedDialog-EN dataset
- No real patient data
- No PHI (Protected Health Information)

### Model Limitations

- LLM may generate plausible but incorrect information
- Hallucination detection is not 100% accurate
- System requires human oversight

## 🔧 Troubleshooting

### Out of Memory

If you encounter OOM errors:

```python
# Use smaller batch size
embedder = TextEmbedder(batch_size=16)

# Use CPU instead of GPU
embedder = TextEmbedder(device="cpu")

# Use smaller chunking token sizes
chunker = FixedLengthChunker(token_size=256)
```

### Slow Performance

- Enable GPU for embedding: `device="cuda"`
- Use HNSW index instead of flat: `index_type="hnsw"`
- Reduce number of dialogues for testing

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📚 Additional Resources

- **MedDialog Dataset:** [Link to dataset]
- **Sentence Transformers:** https://www.sbert.net/
- **FAISS:** https://github.com/facebookresearch/faiss
- **Streamlit:** https://streamlit.io/

## 🤝 Contributing

This is a research project. For modifications:
1. Maintain modular structure
2. Document all changes
3. Update tests
4. Log all experimental parameters

## 📧 Contact

For questions about this research project:
- Email: [your.email@university.edu]
- Research Group: [Lab Name]

## 📄 License

This project is for academic research. Check individual library licenses for dependencies.

---

**Built with:** Python, PyTorch, Transformers, FAISS, Streamlit  
**Purpose:** Academic research on RAG chunking strategies for medical dialogues  
**Status:** Research prototype
