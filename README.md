# RAG Medical Dialogue System

A retrieval-focused system for medical dialogue analysis, designed for academic research comparing chunking strategies.

## 🎯 Project Objective

Build a modular RAG pipeline with evaluation framework and UI to:
- Apply multiple chunking strategies on medical dialogue dataset (112K dialogues)
- Build separate vector indexes for each strategy
- Compare retrieval, hallucination, and efficiency metrics
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
# Chunking strategies for MedDialog — cleaned public copy

This repository provides tools for loading the MedDialog-EN dataset, applying several
chunking strategies, building FAISS vector indexes, and running retrieval experiments.
This public copy has generation/model-serving code removed to keep the repository
lightweight and self-contained for reproducible retrieval experiments.

Key capabilities included in this copy:
- Load MedDialog-EN from a local JSON file (`data/downloaded/meddialog.json`) or from HuggingFace when available.
- Preprocess dialogues with the `preprocessing` utilities.
- Create chunks using multiple strategies implemented in `chunking/`.
- Embed text using the `embeddings.TextEmbedder` (sentence-transformers).
- Build and query FAISS indexes (`vector_store/faiss_index.py`).
- Evaluate retrieval, hallucination checks and efficiency metrics (`evaluation/`).
- A Streamlit UI (`ui/app.py`) that demonstrates retrieval and strategy comparisons. Generation is disabled in the UI.

This README only documents functionality that is implemented in the current codebase.

## Quick setup

1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3. Ensure you have the MedDialog JSON in `data/downloaded/meddialog.json` if you want to run the full dataset. The loader will fall back to sample dialogues if not present.

## Running the Streamlit demo

The UI demonstrates loading a small sample, building indexes for available chunking strategies, and performing retrieval. Generation is intentionally disabled.

```powershell
streamlit run ui/app.py
```

Notes:
- The UI builds FAISS indexes in-memory for each configured strategy; runtime and memory depend on dataset size and `TextEmbedder` settings.

## Programmatic usage

Minimal example (load local file, preprocess, chunk, embed, index, retrieve):

```python
from data.meddialog_loader import MedDialogLoader
from preprocessing.cleaner import TextCleaner
from chunking.fixed import FixedLengthChunker
from embeddings.embedder import TextEmbedder
from vector_store.faiss_index import FAISSIndex
from retrieval.retriever import Retriever

# Load a small sample from local file
loader = MedDialogLoader(dataset_source='local')
dialogs = loader.load_from_local(r"data/downloaded/meddialog.json")

# Preprocess (TextCleaner implements simple cleaning helpers)
cleaner = TextCleaner()
cleaned = cleaner.clean_dialogues(dialogs)

# Chunk (use the fixed chunker as example)
chunker = FixedLengthChunker(token_size=512)
chunks = chunker.chunk_dialogues(cleaned)

# Embed
embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
embeddings = embedder.embed_chunks(chunks)

# Build FAISS index
index = FAISSIndex(embedder.embedding_dim, index_type="flat", metric="cosine")
index.build_index(embeddings, chunks)

# Retrieve
retriever = Retriever(embedder, index)
results, scores = retriever.retrieve("What are the symptoms of chest pain?", top_k=5)
print(results[0])
```

## Notes on removed/disabled features

- The generation pipeline and large LLM integrations are not included in this public copy. The UI and code intentionally avoid importing heavy generation models.
- Some evaluation utilities depend on external libraries (for example, `sentence-transformers` and `datasets`). If you only want to run retrieval demos, use CPU and smaller samples to reduce memory usage.

## Project layout (high-level)

Relevant files and folders:

- `data/` — `meddialog_loader.py` (loader and sample data)
- `preprocessing/` — `cleaner.py` (text cleaning utilities)
- `chunking/` — implementations for several chunking strategies
- `embeddings/` — `embedder.py` (TextEmbedder using sentence-transformers)
- `vector_store/` — `faiss_index.py` (build/search indexes)
- `retrieval/` — `retriever.py` (retrieve and format contexts)
- `evaluation/` — retrieval/hallucination/efficiency helpers
- `ui/app.py` — Streamlit demonstration (generation disabled)

## Troubleshooting

- If `sentence-transformers` or `datasets` cause import errors, install the packages from `requirements.txt` and restart the environment. For quick tests, use the sample dialogues provided by the loader (they do not require downloads).
- If FAISS is not available or fails to build, you can still test chunking and embedding code; FAISS is required for index building and search.

## License & Disclaimer

This code is provided for research purposes only. It is not medical software and must not be used for clinical decision making.

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
