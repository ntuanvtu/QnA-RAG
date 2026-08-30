# Technical Document Q&A Assistant

> Intelligent question-answering system for PDF documents with source citations and hallucination prevention.

## Overview

This is a **Retrieval-Augmented Generation (RAG)** chatbot designed to answer questions about technical PDF documents in Vietnamese and English, with precise source citations and built-in safeguards against hallucination.

**Key Principle:** Rather than letting the language model answer from memory (prone to fabrication), the system **first retrieves relevant document passages**, then asks Claude to answer **exclusively based on those passages** with explicit source references.

**Focus:** This project emphasizes not just functionality but **measurable quality**. It includes a self-built quantitative evaluation suite (35 questions) measuring retrieval accuracy, answer groundedness, out-of-scope refusal capability, and latency.

## Core Features

### 1. **Intelligent Chatbot Interface**
- Upload PDFs and ask questions in a unified chat experience
- Real-time document processing—no separate "upload" step required
- Automatic deduplication: re-uploading the same file is safely skipped

### 2. **Source-Grounded Answers**
- Every answer includes precise source citations: `[source: filename p.X]`
- Users can immediately verify claims against the original document
- Confidence scores displayed for answer reliability

### 3. **Two-Tier Hallucination Prevention**
- **Tier 1 (Retrieval):** If no sufficiently relevant passages are found → decline to answer without calling the LLM (saves cost)
- **Tier 2 (Generation):** If Claude recognizes insufficient context → return refusal rather than guessing

### 4. **Intelligent Query Handling**
- Understands Vietnamese questions on English-language documents
- Automatically rewrites questions into optimized search queries
- Removes instruction phrases ("explain clearly," "summarize," etc.) to improve retrieval
- Clean English queries skip rewriting for efficiency

### 5. **Document Summarization**
- "Summarize this document" or "List all sections" triggers whole-document analysis
- Uses map-reduce strategy over every passage (not just top-4)
- Long documents are sampled uniformly with progress notes
- Perfect for executive summaries or comprehensive overviews

### 6. **Scope Filtering**
- Dropdown to limit questions to a specific document
- Useful when working with multiple PDFs simultaneously

### 7. **Local Embeddings (Free & Private)**
- Embedding and retrieval steps run entirely locally using sentence-transformers
- No API key required for document ingestion or search
- Only the answer generation step requires Anthropic API
- Zero data leakage to external services

### 8. **Quantitative Evaluation**
- Built-in test suite: 35 questions (25 in-scope, 10 out-of-scope)
- Automated grading using Claude as a faithfulness judge
- Measurable metrics: retrieval accuracy, groundedness, refusal precision, latency

## Quick Start

### Prerequisites
- Python 3.9+
- Anthropic API key (get it at https://console.anthropic.com)

### Installation

```bash
# Clone and setup
git clone https://github.com/ntuanvtu/QnA-RAG.git
cd QnA-RAG

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure API key
copy .env.example .env              # Windows
# cp .env.example .env              # Linux/macOS
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run the Web Interface

```bash
uvicorn app.api:app --reload
# Open http://127.0.0.1:8000 in your browser
```

**Usage:**
1. Click **📎** to attach a PDF file
2. Choose scope: all documents or a specific one
3. Check **"Summarize full document"** if needed
4. Type your question and press **Send** (or Enter)
5. Read the answer with source citations and confidence score

### Ingest Documents via CLI (No API Key Required)

```bash
# Place PDF files in the data/ folder, then:
python -m app.ingest data/filename.pdf     # Ingest single file
python -m app.ingest                       # Ingest all PDFs in data/
```

### Quick Question from Terminal

```bash
python -m app.rag "What are the key components of Algorithm X?"
python -m app.rag "summarize"              # Summary mode (if one document is loaded)
```

### Run the Evaluation Suite

```bash
python -m eval.run_eval                    # Reads eval/testset.json, writes eval/results.json
```

### Run Unit Tests

```bash
pytest -q                                  # No API key required (~20s)
```

## System Architecture

### Data Flow

#### **1. Document Ingestion Pipeline**

```
PDF Files (data/ folder)
    ↓
┌────────────────────────────────────────────────────┐
│ app/ingest.py: Extract & Prepare                   │
│ • Read PDF text page-by-page (pypdfium2)           │
│ • Detect sparse PDFs (slide decks): merge pages    │
│ • Split into chunks (LangChain RecursiveTextSplit)│
│ • Content hash: skip duplicates on re-upload       │
└──────────────────┬─────────────────────────────────┘
                   ↓
        ┌──────────────────────────────────┐
        │ app/embeddings.py                │
        │ Convert chunks → embeddings      │
        │ Model: all-MiniLM-L6-v2 (local)  │
        │ 22M params, runs offline         │
        └────────────┬─────────────────────┘
                     ↓
        ┌──────────────────────────────────┐
        │ app/vectorstore.py               │
        │ Store embeddings in Chroma       │
        │ Cosine similarity space          │
        │ Persisted: storage/chroma/       │
        └──────────────────────────────────┘
```

**Key Features:**
- ✅ No API key needed for ingestion
- ✅ Automatic deduplication via content hash
- ✅ Sparse PDF detection: slides with <700 chars/page are merged
- ✅ Configurable chunk size (default: 1000 chars with 150 char overlap)

---

#### **2. Question Answering Pipeline**

```
                User Question
                      ↓
    ┌─────────────────────────────────────┐
    │ app/rag.py: Orchestration Layer     │
    │ Normalize question (NFC unicode)    │
    │ Detect intent: summary vs. regular  │
    └─────────┬───────────────────────────┘
              │
              ├─ Is it a summary request?
              │  ("summarize", "list all", "overview", etc.)
              │
              ├─ YES → Go to SUMMARY BRANCH
              │
              └─ NO → Go to REGULAR BRANCH


                SUMMARY BRANCH:
                      ↓
    ┌────────────────────────────────────┐
    │ app/summarize.py: Map-Reduce       │
    │ • Get ALL chunks from one file     │
    │ • Batch into 6000-char segments    │
    │ • Claude summarizes each batch     │
    │ • Reduce: combine summaries        │
    │ • Long docs: sample uniformly      │
    └──────────────┬─────────────────────┘
                   │ (Summary answer ready)
                   ↓
    ┌────────────────────────────────────┐
    │ Return Answer                      │
    │ • Content: full summary            │
    │ • Confidence: 1.0 (always trusted) │
    │ • Source: filename                 │
    └────────────────────────────────────┘


                REGULAR BRANCH:
                    ↓
    ┌────────────────────────────────────┐
    │ app/rag.py: Query Rewrite          │
    │ Check: Is question in Vietnamese   │
    │        or has instruction phrases? │
    └──────────────┬─────────────────────┘
                   │
                   ├─ YES → Claude rewrites to search query
                   │        Remove filler: "please", "explain"
                   │        Translate to English if needed
                   │
                   └─ NO → Use original question
                          (clean English queries skip rewriting)
                   │
                   ↓
    ┌─────────────────────────────────────┐
    │ app/retriever.py: Search            │
    │ • Find top-4 similar chunks         │
    │ • Cosine similarity scoring (0-1)   │
    │ • Optional: filter by scope         │
    │ • Calculate confidence:             │
    │   confidence = highest_score        │
    └──────────────┬──────────────────────┘
                   │
                   ↓
    ┌────────────────────────────────────┐
    │ CONFIDENCE CHECK (TIER 1 FALLBACK) │
    │ confidence < 0.35?                 │
    └─┬──────────────────────────────────┘
      │
      ├─ YES → FALLBACK (no LLM call, saves cost)
      │        Return: "No relevant information found"
      │        Confidence: 0
      │
      └─ NO → Call Claude (TIER 2 decision)
              │
              ↓
        ┌───────────────────────────────┐
        │ app/llm.py: Claude API Call   │
        │ Prompt structure:             │
        │  • SYSTEM: Be grounded,       │
        │    cite sources, refuse if    │
        │    context insufficient       │
        │  • USER: [CONTEXT] + [Q]      │
        │  • Get: answer + source refs  │
        └──────────┬────────────────────┘
                   │
                   ├─ Claude says: "I cannot answer based on..."
                   │              → FALLBACK (TIER 2)
                   │
                   └─ Claude provides grounded answer
                      → Return with:
                        • Answer text
                        • Source citations [source: file p.X]
                        • Confidence score
                        • Latency timing


FINAL OUTPUT:
    ↓
    ┌────────────────────────────────────┐
    │ API Response (JSON)                │
    │ {                                  │
    │   "answer": "...",                 │
    │   "sources": ["file1", "file2"],   │
    │   "confidence": 0.85,              │
    │   "is_fallback": false,            │
    │   "latency_s": 2.3,                │
    │   "search_query": "rewritten q"    │
    │ }                                  │
    └────────────────────────────────────┘
```

**Two-Tier Hallucination Prevention:**
- **Tier 1 (Retrieval):** Low confidence → reject without LLM (cost-effective)
- **Tier 2 (Generation):** Claude self-recognizes insufficient context → refuse

---

#### **3. Key Decision Points**

| Decision | Logic | Outcome |
|----------|-------|---------|
| **Summary Mode?** | Contains: "summarize", "overview", "list all" | → Map-reduce over ALL chunks |
| **Query Rewrite?** | Vietnamese OR instruction phrases | → Claude optimizes for search |
| **Confidence Threshold** | similarity_score < 0.35 | → FALLBACK Tier-1 (no LLM call) |
| **Context Insufficient?** | Claude detects gaps | → FALLBACK Tier-2 (refuse answer) |

---

#### **4. File Dependencies**

```
User Request
    ↓
app/rag.py (entry point for CLI, API, eval)
    ├─→ app/ingest.py (if PDF attached)
    ├─→ app/summarize.py (if summary mode)
    ├─→ app/retriever.py (always for regular Q)
    │    ├─→ app/vectorstore.py
    │    └─→ app/embeddings.py
    ├─→ app/llm.py (if confidence >= threshold)
    └─→ Returns Answer object
           (answer, sources, confidence, latency_s)
```

### Technology Stack

| Component | Technology | File | Rationale |
|---|---|---|---|
| **API & Web UI** | FastAPI + Vanilla JS | [app/api.py](app/api.py), [app/static/index.html](app/static/index.html) | Zero build step, lightweight, responsive |
| **PDF Processing** | pypdfium2 + LangChain | [app/ingest.py](app/ingest.py) | Reliable PDF extraction, handles presentations |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | [app/embeddings.py](app/embeddings.py) | Free, runs locally, 22M parameters |
| **Vector Database** | Chroma (disk-persisted, cosine) | [app/vectorstore.py](app/vectorstore.py) | Lightweight, no external service needed |
| **Retrieval** | Cosine similarity + confidence scoring | [app/retriever.py](app/retriever.py) | Configurable thresholds, scope filtering |
| **Query Rewriting & Fallback** | Multi-stage pipeline | [app/rag.py](app/rag.py) | Language detection, instruction removal |
| **Document Summarization** | Map-reduce (Claude batching) | [app/summarize.py](app/summarize.py) | Handles long documents efficiently |
| **LLM** | Claude Haiku 4.5 | [app/llm.py](app/llm.py) | Fast, cheap, excellent instruction following |
| **Evaluation** | Self-built test suite + Claude judge | [eval/run_eval.py](eval/run_eval.py) | Automated groundedness scoring |

**Design Philosophy:** Embeddings and retrieval are local (no API cost, no data leakage). Only answer generation calls Claude. LangChain is used as a thin orchestration layer, not a dependency.

## Configuration

All parameters are defined in [config.py](config.py) and can be overridden via environment variables or `.env` file.

**Priority order:** Environment variable > `.env` > Default value

| Parameter | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | **Required.** Get from https://console.anthropic.com |
| `LLM_MODEL` | `claude-haiku-4-5` | Switch to `claude-sonnet-5` for higher quality (slower, more expensive) |
| `CONFIDENCE_THRESHOLD` | `0.35` | Minimum similarity score to avoid Tier-1 fallback |
| `TOP_K` | `4` | Number of document passages to retrieve per question |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Text chunk size when splitting documents |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `SPARSE_MEDIAN_THRESHOLD` | `700` | Characters/page below this → treat as slide deck, merge pages |
| `SPARSE_BLOCK_CHARS` | `600` | Minimum block size when merging sparse slides |
| `ENABLE_QUERY_REWRITE` | `true` | Allow Claude to rewrite non-English questions before retrieval |
| `SUMMARY_BATCH_CHARS` / `SUMMARY_MAX_BATCHES` | `6000` / `10` | Summary mode: batch size and max batches (long docs sampled uniformly) |
| `VECTOR_DIR` | `storage/chroma/` | Path to persistent Chroma database |

### ⚠️ Important: Configuration Changes

When modifying retrieval-related parameters (`CHUNK_SIZE`, `EMBEDDING_MODEL`, etc.):
1. Delete the `storage/chroma/` directory
2. Re-run `python -m app.ingest` to rebuild the vector store

Old embeddings in Chroma are incompatible with new configurations and won't auto-update.

## Evaluation Results

**Test Suite:** 35 questions across 2 sample technical documents
- **25 In-Scope Questions:** Answers exist in the documents
- **10 Out-of-Scope Questions:** Questions about topics not covered (for refusal testing)

**Configuration:**
- Model: `claude-haiku-4-5`
- Embeddings: `all-MiniLM-L6-v2`
- `TOP_K = 4` | `CONFIDENCE_THRESHOLD = 0.35`

**Results:**

| Metric | Score | Interpretation |
|---|---|---|
| **Retrieval Accuracy@4** | **96%** (24/25) | Relevant passages are in the top-4 retrieved documents |
| **Groundedness** | **100%** (23/23) | All answered questions stay faithful to sources (Claude RAGAS judge) |
| **Refusal Precision** | **10/10** | Out-of-scope questions correctly declined (zero false positives) |
| **In-Scope Answer Rate** | **92%** (23/25) | In-scope questions actually answered (vs. false refusals) |
| **Average Latency** | **2.9s** (p95: 6.7s) | Including retrieval + Claude API call time |

**Notes on the 2/25 In-Scope Misses:**
The retriever (MiniLM + fixed chunking) failed to rank the answer-containing passage in top-4. Per design, the system **refuses to guess**—the correct behavior. Quality could be improved with hybrid search or a re-ranker (out of current scope).

**Evaluation Details:** See [eval/results.json](eval/results.json) for per-question breakdowns.

> The test suite contains English questions on English documents. Vietnamese language capability (and query rewriting) is validated through separate manual testing.

## Known Limitations

### Design Constraints (Intentional)

1. **Sparse/Presentation PDFs:** Slide decks with few characters per page are automatically merged into larger blocks (controlled by `SPARSE_MEDIAN_THRESHOLD`). This improves chunking quality but means source citations point to the first slide of a cluster, not the exact slide.

2. **Text in Images:** Content baked into images (diagrams, screenshots, charts as raster) is not extracted. Optical character recognition (OCR) is out of scope. For such questions, the system appropriately declines rather than hallucinating.

3. **Long Document Summarization:** Very long files are sampled uniformly across `SUMMARY_MAX_BATCHES` segments, not summarized in full. Progress notes indicate sampling coverage.

4. **No Conversation Memory:** Each question is processed independently. Multi-turn dialogue like "explain point #2 further" is not supported. Workaround: rephrase the question in full context.

5. **PDF Format Only:** Accepts only PDFs with extractable text (not Word, HTML, or image-only scans).

### Performance Considerations

- **First Run:** Embedding model (~90 MB) is downloaded and cached on first use
- **Cold Start:** Chroma initialization adds ~500ms to the first question
- **Large Batches:** Summarizing 100+ document pages may take 60–90 seconds (map-reduce overhead)

---

## Out of Scope (Intentional Design Decisions)

The following are explicitly out of scope to keep the project focused:

- Agent/tool-calling and multi-step reasoning
- Conversational memory and follow-up context
- Optical character recognition (OCR) for images
- Hybrid search, dense-retrieval re-ranking
- Multi-language query expansion beyond Vietnamese/English
- Domain-specific handling (legal, medical, financial constraints)
- Production-scale deployment (load testing, distributed indexing)
- Streaming responses, real-time collaboration

---

## Project Structure

```
app/
  ├── ingest.py           Read PDF → merge sparse slides → chunk → embed → save to Chroma
  ├── embeddings.py       Local embedding model (singleton)
  ├── vectorstore.py      Chroma integration + document listing
  ├── retriever.py        Retrieval top-k + confidence scoring + scope filtering
  ├── rag.py              Orchestration: rewrite query, route summary, call Claude, fallback
  ├── summarize.py        Whole-document summarization (map-reduce)
  ├── llm.py              Claude client initialization (singleton)
  ├── api.py              FastAPI: chatbot UI + endpoints (/api/chat, /api/reset, etc.)
  └── static/
      └── index.html      Modern chat interface with source panel

eval/
  ├── testset.json        35 questions (in-scope + out-of-scope)
  └── run_eval.py         Run full test suite, auto-grade, write results.json

tests/
  ├── test_pipeline.py    Unit tests (no API key required)
  └── test_ui_playwright.py   Real browser tests with Playwright

config.py                 All configuration parameters (pydantic-settings)
conftest.py              pytest fixtures and setup
requirements.txt         Python dependencies

storage/
  └── chroma/             Persisted vector database (created after first ingest)

data/
  └── (place your PDFs here)
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError" or import errors

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

Ensure you're in the correct virtual environment:
```bash
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate    # Linux/macOS
```

### Issue: Empty vector store / No documents ingested

**Solution:**
1. Verify PDF files are in the `data/` directory
2. Run `python -m app.ingest` manually to test
3. Check `storage/chroma/` exists and has content
4. Check console for error messages during ingestion

### Issue: Slow retrieval or low quality answers

**Solution:**
- Reduce `TOP_K` from 4 to 3 (faster)
- Increase `TOP_K` from 4 to 6 (more context, slower)
- Lower `CONFIDENCE_THRESHOLD` from 0.35 to 0.25 (more answers, riskier)
- Experiment with `CHUNK_SIZE` (larger chunks for more context, smaller for precision)

### Issue: "Invalid API key" error

**Solution:**
1. Verify `ANTHROPIC_API_KEY` is set in `.env`
2. Check it's a valid key from https://console.anthropic.com
3. Ensure no extra whitespace in `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-v0-xxxxx  # Correct
   ANTHROPIC_API_KEY=sk-ant-v0-xxxxx   # Extra space = wrong
   ```

### Issue: Chroma database corruption after config change

**Solution:**
```bash
rmdir /s /q storage\chroma\    # Windows
# rm -rf storage/chroma/       # Linux/macOS
python -m app.ingest           # Rebuild
```

---

## Development Workflow

### Adding a New Test Question

1. Add the question and expected keywords to [eval/testset.json](eval/testset.json)
2. Verify retrieval first:
   ```python
   python
   >>> from app.retriever import retrieve
   >>> retrieve("your question")
   ```
3. Run full evaluation: `python -m eval.run_eval`
4. Commit with clear message about the test case

### Modifying RAG Pipeline

Before committing changes to `app/rag.py`, `app/retriever.py`, or similar:
1. Run unit tests: `pytest -q`
2. Run browser tests: `pytest tests/test_ui_playwright.py -q`
3. Spot-check with manual queries: `python -m app.rag "test question"`

### Performance Profiling

For latency analysis:
```python
python -m app.rag "your question"   # Includes timing in output
```

To profile retrieval specifically, check the `latency_s` field in API responses.

---

## Contributing

Contributions are welcome! Please:

1. **Fork and clone** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** and ensure tests pass:
   ```bash
   pytest -q
   ```
4. **Commit with clear messages** (see [CLAUDE.md](CLAUDE.md) for style guidelines)
5. **Push and create a Pull Request**

---

## Citation & License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

**If you use this in research or production, please cite:**

```bibtex
@software{qna_rag_2024,
  title={Technical Document Q&A Assistant: RAG with Two-Tier Fallback},
  author={Your Name},
  year={2024},
  url={https://github.com/ntuanvtu/QnA-RAG}
}
```

---

## Contact & Support

- **Issues:** Open a GitHub issue with reproduction steps
- **Discussions:** Use GitHub Discussions for feature requests and questions
- **Email:** Contact via GitHub profile

---

## Acknowledgments

- **Claude (Anthropic):** LLM for answer generation and faithfulness evaluation
- **Sentence-Transformers (SBERT):** Efficient embedding model
- **Chroma:** Lightweight vector database
- **LangChain:** Text splitting and LLM integration utilities
- **Playwright:** Real browser automation for testing

---

**Last Updated:** August 30, 2024  
**Version:** 1.0.0

## Known Limitations

### Design Constraints (Intentional)

1. **Sparse/Presentation PDFs:** Slide decks with few characters per page are automatically merged into larger blocks (controlled by `SPARSE_MEDIAN_THRESHOLD`). This improves chunking quality but means source citations point to the first slide of a cluster, not the exact slide.

2. **Text in Images:** Content baked into images (diagrams, screenshots, charts as raster) is not extracted. Optical character recognition (OCR) is out of scope. For such questions, the system appropriately declines rather than hallucinating.

3. **Long Document Summarization:** Very long files are sampled uniformly across `SUMMARY_MAX_BATCHES` segments, not summarized in full. Progress notes indicate sampling coverage.

4. **No Conversation Memory:** Each question is processed independently. Multi-turn dialogue like "explain point #2 further" is not supported. Workaround: rephrase the question in full context.

5. **PDF Format Only:** Accepts only PDFs with extractable text (not Word, HTML, or image-only scans).

### Performance Considerations

- **First Run:** Embedding model (~90 MB) is downloaded and cached on first use
- **Cold Start:** Chroma initialization adds ~500ms to the first question
- **Large Batches:** Summarizing 100+ document pages may take 60–90 seconds (map-reduce overhead)

## Out of Scope (Intentional Design Decisions)

The following are explicitly out of scope to keep the project focused:

- Agent/tool-calling and multi-step reasoning
- Conversational memory and follow-up context
- Optical character recognition (OCR) for images
- Hybrid search, dense-retrieval re-ranking
- Multi-language query expansion beyond Vietnamese/English
- Domain-specific handling (legal, medical, financial constraints)
- Production-scale deployment (load testing, distributed indexing)
- Streaming responses, real-time collaboration

---

## Project Structure

```
app/
  ├── ingest.py           Read PDF → merge sparse slides → chunk → embed → save to Chroma
  ├── embeddings.py       Local embedding model (singleton)
  ├── vectorstore.py      Chroma integration + document listing
  ├── retriever.py        Retrieval top-k + confidence scoring + scope filtering
  ├── rag.py              Orchestration: rewrite query, route summary, call Claude, fallback
  ├── summarize.py        Whole-document summarization (map-reduce)
  ├── llm.py              Claude client initialization (singleton)
  ├── api.py              FastAPI: chatbot UI + endpoints (/api/chat, /api/reset, etc.)
  └── static/
      └── index.html      Modern chat interface with source panel

eval/
  ├── testset.json        35 questions (in-scope + out-of-scope)
  └── run_eval.py         Run full test suite, auto-grade, write results.json

tests/
  ├── test_pipeline.py    Unit tests (no API key required)
  └── test_ui_playwright.py   Real browser tests with Playwright

config.py                 All configuration parameters (pydantic-settings)
conftest.py              pytest fixtures and setup
requirements.txt         Python dependencies

storage/
  └── chroma/             Persisted vector database (created after first ingest)

data/
  └── (place your PDFs here)
```
