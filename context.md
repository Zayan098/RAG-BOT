# AI Agent Factory — Book Chatbot (Project Context)

## Goal
A free, open-source RAG chatbot answering questions about **The AI Agent Factory** book (Panaversity, https://agentfactory.panaversity.org/), grounded only in the book's content, with chapter citations. $0 recurring cost (no paid APIs/hosting).

## Architecture
```
agentfactory.panaversity.org/sitemap.xml (81 English doc pages)
  → scrape_book.py: fetch SSR HTML → extract <article> → markdownify → book_content/*.md + manifest.json
  → ingest.py: MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter (800/120)
  → sentence-transformers all-MiniLM-L6-v2 (local CPU, normalized) → ChromaDB (cosine)
  → query.py: top-k retrieval → Groq LLM (groq/compound-mini, fallback groq/compound) → answer + citations
  (dual-mode prompt: book questions grounded in excerpts, general questions answered normally; no URLs in answers)
  → app.py: Streamlit chat UI
```

## ⚠️ Critical finding (deviates from original plan.md)
`github.com/panaversity/agentfactory` **no longer exists** (404 — deleted/private). The plan's "git clone the source repo" phase is impossible. We chose **live-site scraping** (current, canonical, real citation URLs). A stale fork (`Cypherpunk-Labs/panaversity-agentfactory`, Feb 2026) exists but is unused.

## Current status
| Phase | Status |
|---|---|
| 1. Scaffold + scraper | ✅ Done — 81 pages, 6.3 MB markdown, `book_content/manifest.json` (url/title/section per page) |
| 2. Ingestion | ✅ Done — 12,922 chunks in `chroma_db/`, collection `book`, cosine + normalized embeddings |
| 3. query.py (retrieval + Groq) | ✅ Done — CLI tested: in-book answers w/ citations, out-of-scope guardrail, fallback + retry on 413/429 |
| 4. app.py (Streamlit UI) | ✅ Done — chat UI, session history, clickable sources, missing-key warning |
| 4.5 Answer-quality pass | ✅ Done — dual-mode prompt (book vs general), conversation-aware reformulation, no link salad (URLs stripped from context), per-chunk 0.5 filter, greeting handling, quota-exhausted + rate-limit messages |
| 5. Evaluation (15–20 questions) | ✅ Done — 27 questions in `eval_questions.md`, `eval.py` dry-run scores mode + hygiene; 27/27 PASS (LLM scoring pending quota) |
| 6. README/plan.md update | ✅ Done — `README.md` reflects scraping reality (no repo clone); plan.md never existed on disk |

## Environment (Windows)
- Python **3.14.6** via `py -3.14` (bare `python` resolves to a dead WindowsApps stub — always use `venv\Scripts\python.exe`)
- venv: `D:\RAGBOT\venv` — torch 2.13.0+cpu, sentence-transformers 5.7.0, chromadb 1.5.9, streamlit 1.61.1, groq, langchain-text-splitters
- `GROQ_API_KEY`: **set** in `.streamlit/secrets.toml` (user's key, on_demand tier)
- torchvision 0.28.0+cpu installed (2026-08-18) — fixes transformers 5.x zoedepth import crash in some contexts

## Key decisions
- **Embedding model**: `all-MiniLM-L6-v2` (384-dim, CPU) — change in `ingest.py` + `query.py` constants if needed
- **Chunking**: 800 chars / 120 overlap, split on `#`–`###` headers; H2/H3 prepended to chunk text
- **Retrieval**: top-k 6, max 2 chunks/source; per-chunk filter `MAX_CHUNK_DISTANCE=0.5` (verified: in-book ~0.16–0.45, "weather in Lahore" ~0.65+). 0 hits → **general mode** (answered like a regular assistant, no book context)
- **LLM**: Groq `groq/compound-mini` primary (internally routes to `llama-3.3-70b-versatile` — verified via its rate-limit error), fallback `groq/compound`. Free tier ≈ 30K TPM / **100K TPD** (daily cap on the 70b backend — hit during testing 2026-08-18). Retry w/ backoff (3×, sleeps capped at 30s) on 413/429. `tools=[]` + `citation_options="disabled"` keeps it grounded.
- **Prompt**: user-specified dual-mode spec — book questions taught from excerpts only, general questions answered normally; no URLs/links/chunk jargon in answers; greeting handled locally (no quota)
- **Conversation memory**: follow-up questions are rewritten into standalone questions via a small Groq call (`reformulate()`, temp 0, max 80 tokens) using the last 6 turns; answers include last 4 turns as context

## Gotchas learned (don't repeat)
1. **ChromaDB batching bug**: flush must accumulate embeddings too (ids/documents/metadatas/embeddings all flush together) — caused first ingest crash
2. **ChromaDB default metric is L2** — must set `metadata={"hnsw:space": "cosine"}` + normalize embeddings
3. **Use raw chromadb client**, not langchain-community wrapper (version conflicts with chromadb 1.x)
4. **Windows console cp1252**: CLI prints of unicode (`←`, em-dashes) crash — `query.py` must `sys.stdout.reconfigure(encoding="utf-8")`
5. **PowerShell**: `.TrimEnd('.md')` trims chars, not suffix; `python` ≠ `py -3.14`; nested quotes in `-c` strings break — prefer script files
6. **HF symlink warning** (cache without Developer Mode) — harmless
7. Some pages are low-information boilerplate (e.g. `ecosystem/system-of-record` is MCP setup instructions) — retrieval correctly skips them
8. `.streamlit/secrets.toml`, `chroma_db/`, `book_content/` are gitignored
9. **Groq model lineup changed (2026)**: old `llama-3.3-70b-versatile`/`llama-3.1-8b-instant` IDs are gone; compound IDs route internally (compound-mini → 70b-versatile). `groq/compound` currently returns 413 on RAG-shaped prompts — keep it fallback-only
10. Groq free tier is TPM-limited (~30K/min) and **TPD-limited (100K/day on the 70b backend)**; rapid-fire testing exhausts the daily cap — retries + backoff required; app shows a "quota exhausted, try tomorrow" message
11. **all-MiniLM ranks fuzzy questions weakly** (0.41–0.56 for structural questions vs 0.16–0.33 clean ones) — fixed via per-chunk filter `MAX_CHUNK_DISTANCE=0.5` + `MIN_HITS=1`, and LLM-based query reformulation for follow-ups; 0.45 was too strict (killed "what are you capable of")
12. **Compound router sends calls to different sub-models with separate limits** (gpt-oss-120b: 8K TPM, 70b-versatile: 12K TPM + 100K TPD) — quota errors may name either
13. **No links/URLs in answers**: the model hyperlinks headings when allowed → dual-mode `SYSTEM_PROMPT` forbids URLs/paths/chunk jargon entirely; context chunks carry only `[Title | Section]` labels (URLs live in metadata for the UI Sources list)
14. **Retry-After can be huge (45 min)** — `_retry_delay` caps sleeps at 30s; TPM failures get a friendly "try again in a moment" message
15. **Free tier also caps requests per day (RPD 250 on compound-mini, rolling window)** — "requests per day" errors now skip retries and show the quota-exhausted message immediately

## Commands
```
venv\Scripts\python.exe scrape_book.py [--refresh]   # refresh content cache
venv\Scripts\python.exe ingest.py [--rebuild]        # rebuild vector store (~12 min CPU)
venv\Scripts\python.exe query.py "question"          # Phase 3 (CLI test)
venv\Scripts\streamlit run app.py                    # Phase 4 (UI)
```

## Next steps
1. `query.py` — retrieval + prompt guardrails + Groq call w/ fallback + threshold + utf-8 fix ✅
2. `app.py` — chat UI (session history, citations, missing-key screen) ✅
3. `eval_questions.md` + `eval.py` — 27 questions incl. out-of-book traps ✅
4. README/plan.md update ✅
5. Optional: re-run `eval.py --llm` when the RPD window resets for end-to-end answer scoring