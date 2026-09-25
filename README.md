 # AI RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions from a company handbook — grounded in real document text, semantically cached, checked for hallucinations, and routed through a resilient LLM gateway with cost tracking.

Built as a hands-on learning project covering the core concepts behind production AI engineering: embeddings, vector search, guardrails, semantic caching, LLM gateways, conversation memory, agentic tool-calling, and automated evals.

## Features

- **Grounded Q&A** — answers are generated only from retrieved document context, not model memory
- **Semantic caching** — near-duplicate questions skip the LLM call entirely, using a combined similarity + chunk-overlap check
- **Guardrails** — a cheap rule-based check plus an LLM-as-judge groundedness check catch ungrounded or hallucinated answers
- **LLM gateway** — rate limiting, ordered fallback across models, and real dollar-cost tracking per call
- **Conversation memory** — a sliding window of recent turns so follow-up questions ("can I carry *those* over?") resolve correctly
- **Agentic tool-calling** — the model can call real functions (e.g. date lookups, business-day calculations) when the question needs computation the documents can't answer
- **Automated evals** — a hand-written test set scored automatically against the full live pipeline

## Architecture

```
data/                   Source documents (.txt)
chroma_db/              Persisted vector store + fitted embedder
src/
  embeddings.py         TfidfEmbedder + NeuralEmbedder (swappable, same interface)
  ingest.py             Chunk documents (sentence-aware) → embed → store
  retrieve.py           Question → nearest chunks (semantic search)
  generate.py           Chunks + memory → grounded answer, via the gateway
  guardrails.py         Rule-based + LLM-as-judge groundedness checks
  cache.py              Semantic cache (similarity + chunk-overlap match)
  gateway.py            Rate limiting, model fallback, cost tracking
  memory.py             Sliding-window conversation history
  agent.py              Tool-calling (get_current_date, business_days_until)
  main.py               Ties everything together into a chat loop
tests/
  eval_dataset.py        Hand-written question/expected-answer test cases
  eval_runner.py          Runs the eval set through the live pipeline, scores it
```

## How it works

1. **Ingestion** (`ingest.py`, run once or whenever documents change): documents are split into overlapping, sentence-aware chunks, embedded with a neural model (`all-MiniLM-L6-v2`), and stored in a local Chroma vector database.
2. **On each question** (`main.py`): retrieve the nearest chunks → check the semantic cache → if a miss, generate a grounded answer through the LLM gateway → run guardrail checks → cache and remember the result.

## Setup

```bash
python -m venv rag_env
source rag_env/bin/activate
pip install groq chromadb scikit-learn sentence-transformers python-dotenv
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Add your source documents to `data/` as `.txt` files, then run ingestion:

```bash
cd src
python ingest.py
```

Start the assistant:

```bash
python main.py
```

Type a question, or `quit` to exit (prints a session cost summary on exit).

## Running the evals

```bash
python tests/eval_runner.py
```

Runs every case in `eval_dataset.py` through the live pipeline and prints a pass/fail report plus an aggregate score.

**Current score: 5/6 (83%)**

## Known limitations

- **Chunk-level cache granularity**: the semantic cache checks whether two questions draw from the same retrieved chunks. When two different facts sit close together in the source text and land in the same chunk, questions about either fact can be mistaken for duplicates. A future fix would use finer-grained, one-fact-per-chunk splitting or a fact-level cache key instead of chunk-level.
- **TF-IDF embedder available but not default**: `embeddings.py` includes a TF-IDF embedder as a fallback for offline/no-download environments. It cannot recognize synonyms (e.g. "vacation" vs "PTO") since it only matches literal shared vocabulary — use `NeuralEmbedder` whenever network access allows.
- **Small eval set**: 6 hand-written cases is enough to catch systemic bugs (as it did during development) but is not a statistically robust benchmark. A production system would expand this to dozens of cases across more edge cases and document types.
- **Single-document corpus**: currently ingests one sample handbook; ingestion already supports multiple `.txt` files in `data/`, but retrieval quality at larger scale (hundreds of documents) hasn't been tested.

## What this project demonstrates

Built sprint-by-sprint, debugging real issues as they came up rather than following pre-written working code:

- Library API version mismatches (chromadb's embedding-function interface)
- Python indentation bugs silently changing class structure
- Circular imports between modules
- Calibrating thresholds empirically instead of guessing
- Diagnosing a parameter interaction (chunk size × `top_k`) that shifted system behavior in an unexpected direction
- Using an eval suite to catch a regression that manual spot-testing missed entirely