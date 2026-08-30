# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

### Mandatory Core Behavioral Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

#### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

#### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

#### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

#### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### Project Context

A Q&A system over technical PDF documents using a RAG architecture (retrieve → generate), with two-tier fallback against hallucination and a self-built eval suite measuring retrieval accuracy, groundedness, out-of-scope refusal, and latency. See [README.md](README.md) for full context and the latest eval numbers.

#### 1. Common Commands

```bash
# Setup (one time)
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell on Windows
pip install -r requirements.txt
copy .env.example .env              # then fill in ANTHROPIC_API_KEY

# Ingest PDFs into the vector store (no API key needed)
python -m app.ingest                          # every PDF in data/
python -m app.ingest data/some-file.pdf       # a single file

# Ask a quick question from the terminal (needs an API key)
python -m app.rag "your question"

# Run the web server (FastAPI, upload + Q&A UI)
uvicorn app.api:app --reload        # http://127.0.0.1:8000

# Tests
pytest -q                                                          # whole suite, no API key needed
pytest tests/test_pipeline.py::test_fallback_threshold_logic -q    # single test, no PDF/key needed
pytest tests/test_pipeline.py::test_ingest_and_retrieve -q         # needs a PDF in data/

# Quantitative eval (needs an API key, calls Claude ~2x per question -> small cost)
python -m eval.run_eval             # reads eval/testset.json, writes eval/results.json
```

No build step (plain Python). No linter/formatter is configured yet (no ruff/black/flake8) — if you add one, keep it to the existing style rather than auto-formatting the whole repo in the same change.

#### 2. Architecture

Main data flow (see also the diagram in the README):

```
PDF → app/ingest.py (pypdfium2 reads pages → sparse-slide docs get pages merged → RecursiveCharacterTextSplitter chunks them)
    → app/embeddings.py (sentence-transformers, local, no cost)
    → app/vectorstore.py (Chroma, persisted to storage/chroma/, cosine space)

question → app/rag.py answer_question()
         → NFC-normalize, then route:
           • summary intent (or force_summary) → app/summarize.py (map-reduce over ALL chunks of one file)
           • else → _rewrite_query() if _needs_rewrite() (non-English or has an instruction phrase)
                  → app/retriever.py (similarity_search_with_relevance_scores, top_k, optional source filter)
                  → confidence = highest score among top-k
                  → score < CONFIDENCE_THRESHOLD ? tier-1 FALLBACK (LLM never called)
                                                  : app/llm.py (Claude) → answer + [source: file p.X]
                                                    → LLM admits it lacks info? tier-2 FALLBACK
```

- **config.py**: every tunable parameter (model, thresholds, top_k, chunk size...) lives here, read through `pydantic-settings` (precedence: env var > `.env` > default). Don't hardcode parameters elsewhere in `app/`.
- **app/llm.py, app/embeddings.py**: lazy singletons (module-level `_llm`/`_embeddings` + a `get_llm()`/`get_embeddings()` accessor) so the client/model isn't reconstructed on every call.
- **app/retriever.py**: `RetrievalResult.confidence`/`is_confident` is the single source of truth for the tier-1 fallback decision — don't recompute the threshold check elsewhere.
- **app/rag.py**: `answer_question()` is the single entry point (CLI `__main__`, API, eval). It NFC-normalizes the question (HTTP input can be NFD → breaks Vietnamese regex matching), routes summary intent to `app/summarize.py`, optionally rewrites the retrieval query, assembles the prompt, calls the LLM, and detects tier-2 fallback (first 35 chars of `FALLBACK_MESSAGE`). `_needs_rewrite()` gates `_rewrite_query()` — clean English questions skip it (rewriting them shifts retrieval and hurt groundedness in eval). Change answer-generation logic here only.
- **app/summarize.py**: whole-document mode. `summarize_document(source)` pulls every chunk of one file (`where={"source": …}`), batches them (`summary_batch_chars`), map-summarizes each batch, then reduce-combines. Long docs: evenly-sampled down to `summary_max_batches` with a "based on N spread-out parts" note. Returns the same `Answer` dataclass (`confidence=1.0`, no retrieval).
- **app/api.py**: FastAPI. Serves a static chat page (`app/static/index.html`, vanilla JS — no template engine, no build step) and JSON endpoints: `POST /api/chat` (multipart `question` + optional `source` scope + `summarize=1` + `file`), `POST /api/reset` (wipes all docs), `GET /api/files` (ingested sources + chunk counts, for the scope dropdown), `GET /api/info`. No state beyond the on-disk vector store; chat history lives only in the browser.
- **eval/run_eval.py**: runs all of `eval/testset.json` through `answer_question()`, self-grades retrieval hits (`_retrieval_hit` reads `ans.chunks` directly so it reflects the rewritten query) and groundedness (Claude as a RAGAS-style faithfulness judge), writes `eval/results.json`.
- **Page numbering**: `pypdfium2` numbers pages from 0; `ingest.py` adds 1 before storing the `page` metadata — every other place (`retriever.py`, `rag.py`, `eval/testset.json`) uses that already-incremented, human-readable page number.
- **Sparse / presentation PDFs**: if the median chars-per-page is below `sparse_median_threshold` (700), `_merge_sparse_pages()` glues consecutive pages into ≥`sparse_block_chars` (600) blocks before chunking, so slide decks don't become a pile of tiny noisy chunks. Trade-off: a merged block's `page` is the first page of the run, so citations for these docs point at the start of the relevant cluster, not the exact slide. Dense docs (books) skip this untouched.
- **Image-only content**: text baked into raster images (schematics, screenshots) is not extracted — no OCR (out of scope). Pages with no extractable text are silently dropped in `_load_pages`.
- **Ingest dedup**: `ingest_pdf()` hashes the file bytes (SHA-256, stored as `content_hash` in every chunk's metadata). Identical bytes already in the store → skipped, returns `0`. Same filename but different bytes → old chunks (`where={"source": name}`) are deleted first, then the new ones added. So the UI can safely re-ingest on every submit. `python -m eval.run_eval` / config changes still need a clean `storage/chroma/` when `embedding_model` changes (old vectors incompatible).

#### 3. Code Style

- Docstrings/comments are in Vietnamese (explaining *why*, not restating the code); function/variable/class names are in English, `snake_case` for functions/variables, `PascalCase` for dataclasses.
- Every file starts with `from __future__ import annotations` and uses modern type hints (`str | None`, `list[str]`), not `Optional`/`List` from `typing`.
- Use `@dataclass` for plain data structures (`Answer`, `RetrievedChunk`, `RetrievalResult`) — not free-form `dict`s or a Pydantic model for these internal structs.
- Each file under `app/` has a single responsibility (ingest / embeddings / vectorstore / retriever / llm / rag / api) — don't fold another tier's logic into one file.
- Main libraries: **FastAPI** (API/UI), **LangChain** (`langchain-anthropic`, `langchain-huggingface`, `langchain-chroma`, `langchain-text-splitters`) as a thin orchestration layer, **chromadb** (vector DB), **sentence-transformers** (local embeddings), **pypdfium2** (PDF text extraction), **pydantic-settings** (config), **pytest** (tests).
- End-user-facing errors (in `app/api.py`) are caught with `except Exception` and shown to the user — this is intentional (an endpoint must not crash on one bad request), not an oversight to "fix" into a narrower except.

#### 4. Workflow

- After changing code under `app/`, always run `pytest -q` first (no API key needed, ~20s since it loads the embedding model on first run). An autouse fixture in `conftest.py` points `settings.vector_dir` at a tmp dir, so tests never touch the real `storage/chroma/`.
- If a change affects retrieval or fallback behavior (`CONFIDENCE_THRESHOLD`, `top_k`, chunking, or `embedding_model`), you must **delete `storage/chroma/` and re-run `python -m app.ingest`** — existing chunks in Chroma don't auto-update with new config, and embeddings from a different model are incompatible with old ones.
- To check whether a change regresses quality, re-run `python -m eval.run_eval` and compare against the numbers already recorded in the README (`Kết quả eval`) / the previous `eval/results.json`.
- When adding a question to `eval/testset.json`: confirm `relevant_keywords`/`relevant_pages` by calling `retrieve(question)` in a REPL first — don't guess the page number, the retriever doesn't always return the page you'd expect.
- `.env`, `storage/`, and `data/*.pdf` are all in `.gitignore` — don't commit these (see `data/.gitkeep` for why the otherwise-empty `data/` directory is still tracked).
- The repo uses CRLF on Windows; the `LF will be replaced by CRLF` warning on `git add` is expected, not an error.
