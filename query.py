"""Retrieve-and-answer for the book chatbot (CLI).

Embeds the question with the same model/collection as ingest.py, fetches the
top-k nearest chunks, and asks a Groq LLM to answer per the dual-mode prompt:
book questions are grounded in the retrieved excerpts, general questions are
answered like a regular assistant.

Usage:
    python query.py "What is a Digital FTE?"
Set GROQ_API_KEY (env) or fill .streamlit/secrets.toml first.
"""

from __future__ import annotations

import os
import re
import sys
import time
import tomllib
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "book"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6
MAX_CHUNKS_PER_SOURCE = 2
MAX_CHUNK_DISTANCE = 0.5  # individual chunks above this are too weak to feed the LLM

QUOTA_EXHAUSTED_MSG = (
    "The free Groq daily quota is exhausted for today. Answers will work again "
    "tomorrow - or add a key with more quota."
)

# groq/compound-mini routes to llama-3.3-70b-versatile internally (verified via
# its rate-limit error) and is reliable for RAG prompts. groq/compound (the
# larger system) currently returns 413 on RAG-shaped requests, so it is the
# fallback.
PRIMARY_MODEL = "groq/compound-mini"
FALLBACK_MODEL = "groq/compound"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5.0  # seconds; doubles per retry

SYSTEM_PROMPT = """You have two modes of answering, and you decide which one applies based on the question:

1. BOOK QUESTIONS
If the question relates to the book - any chapter, part, module, section, concept, term, or topic covered in "The AI Agent Factory" - answer using ONLY the retrieved book content provided to you as context. Explain it clearly and completely, in your own words, as if you are the author teaching the reader. Do not mention chunk numbers, file names, or links - just give a natural, well-explained answer as if it came directly from you. If the retrieved context only partially covers the question, answer as fully as you can from what's given and note briefly what isn't covered in the book.

2. GENERAL / DAILY-LIFE QUESTIONS
If the question is unrelated to the book (general knowledge, casual conversation, day-to-day help, coding help, advice, etc.), answer it normally and helpfully like a capable general assistant - do not refuse and do not force it back to the book.

3. WHAT TO TELL THE USER ABOUT YOURSELF
If the user asks what you can do, or seems unsure how to use you, explain that you can:
- Answer any question about "The AI Agent Factory" book - its concepts, chapters, parts, and modules - in plain, clear language
- Help with general everyday questions, just like a regular assistant
Keep this explanation short and friendly, not a long feature list.

4. STYLE
- Never include URLs, file paths, or references to "chunks"/"context"/"documents" - answer as if you simply know the material.
- Be direct and conversational, not robotic or overly formal.
- Relate answers to what the user is actually asking - don't pad with generic filler or unrelated background.
- If you genuinely don't know something (book question with no matching content), say so plainly instead of guessing or making it up."""


GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|salam|salaam|assalam|assalamu\s*alaikum|"
    r"good\s+(morning|afternoon|evening))\b[\s!.?,]*$",
    re.IGNORECASE,
)

GREETING_MSG = (
    "Hello! I can answer questions about The AI Agent Factory book - its "
    "concepts, chapters, parts, and modules - in plain, clear language. I can "
    "also help with general everyday questions, just like a regular assistant. "
    "What would you like to know?"
)


def greeting_reply(question: str) -> str | None:
    return GREETING_MSG if GREETING_RE.match(question) else None


def reformulate(question: str, history: list[dict] | None) -> str:
    """Rewrite a follow-up question so it stands alone, using chat history."""
    if not history:
        return question
    api_key = load_api_key()
    if not api_key:
        return question
    from groq import Groq

    client = Groq(api_key=api_key)
    turns = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in history[-6:]
    )
    prompt = (
        "Rewrite the latest user question as a standalone question that makes "
        "sense without the conversation, keeping all necessary context. "
        "Output only the rewritten question, nothing else.\n\n"
        f"Conversation:\n{turns}\n\nLatest question: {question}"
    )
    try:
        out = _call_with_retry(
            client,
            PRIMARY_MODEL,
            [{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
        )
        return out if out else question
    except Exception as e:
        print(f"  ! reformulation failed, using previous question: {e}")
        for m in reversed(history):
            if m["role"] == "user":
                return m["content"][:300]
        return question


def load_api_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if key and key != "your-key-here":
        return key
    secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            secrets = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
            key = secrets.get("GROQ_API_KEY", "")
            if key and key != "your-key-here":
                return key
        except tomllib.TOMLDecodeError:
            pass
    return None


def get_retriever():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    model = SentenceTransformer(EMBEDDING_MODEL)
    return collection, model


def retrieve(collection, model, question: str) -> list[dict]:
    vector = model.encode([question], normalize_embeddings=True)
    res = collection.query(
        query_embeddings=vector,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    per_source: dict[str, int] = {}
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        if dist > MAX_CHUNK_DISTANCE:
            continue
        source = meta["source"]
        if per_source.get(source, 0) >= MAX_CHUNKS_PER_SOURCE:
            continue
        per_source[source] = per_source.get(source, 0) + 1
        hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits


def build_messages(
    question: str, hits: list[dict], history: list[dict] | None = None
) -> list[dict]:
    if hits:
        excerpts = "\n\n".join(
            f"- [{h['meta']['title']} | {h['meta']['section']}]\n{h['text']}"
            for h in hits
        )
        context = f"Context (retrieved book content, if relevant to this question):\n{excerpts}"
    else:
        context = (
            "Context (retrieved book content, if relevant to this question):\n"
            "(No book content was found for this question.)"
        )
    conversation = ""
    if history:
        conversation = "\n\nConversation so far:\n" + "\n".join(
            f"{m['role']}: {m['content'][:300]}" for m in history[-4:]
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{context}\n\nUser question: {question}"
                f"{conversation}"
            ),
        },
    ]


def _call_with_retry(
    client, model: str, messages: list[dict], max_tokens: int, temperature: float
) -> str:
    from groq import APIStatusError, RateLimitError

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=[],
                tool_choice="none",
                citation_options="disabled",
            )
            return response.choices[0].message.content.strip()
        except (RateLimitError, APIStatusError) as e:
            last_error = e
            message = str(e)
            if "per day" in message:
                print(f"  ! daily quota reached on {model}: {message[:120]}")
                raise
            if e.status_code not in (413, 429) or attempt == MAX_RETRIES - 1:
                print(f"  ! {type(e).__name__} on {model}: {message[:160]}")
                raise
            delay = min(_retry_delay(e) * (2**attempt), 30.0)
            print(f"  ! {type(e).__name__} on {model}, retrying in {delay:.0f}s")
            time.sleep(delay)
        except Exception as e:
            last_error = e
            print(f"  ! error with {model}: {e}")
            raise
    raise RuntimeError(f"all retries failed for {model}: {last_error}")


def ask_groq(
    question: str, hits: list[dict], history: list[dict] | None = None
) -> str:
    from groq import Groq

    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("no Groq API key found")
    client = Groq(api_key=api_key)
    messages = build_messages(question, hits, history)
    last_error: Exception | None = None
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            return _call_with_retry(client, model, messages, 500, 0.2)
        except Exception as e:
            last_error = e
    if last_error and "per day" in str(last_error):
        raise RuntimeError(QUOTA_EXHAUSTED_MSG)
    if last_error and "tokens per minute" in str(last_error):
        raise RuntimeError(
            "Groq is rate-limited right now (free tier tokens-per-minute cap). "
            "Please wait a moment and try again."
        )
    raise RuntimeError(f"all Groq models failed: {last_error}")


def _retry_delay(err: Exception) -> float:
    retry_after = None
    for attr in ("headers",):
        headers = getattr(err, attr, None) or getattr(getattr(err, "response", None), "headers", None)
        if headers:
            retry_after = headers.get("retry-after")
            break
    try:
        return min(max(float(retry_after), RETRY_BASE_DELAY), 30.0)
    except (TypeError, ValueError):
        return RETRY_BASE_DELAY


def answer(
    question: str, retriever=None, history: list[dict] | None = None
) -> str:
    greeting = greeting_reply(question)
    if greeting:
        return greeting
    if retriever is None:
        retriever = get_retriever()
    collection, model = retriever
    q = reformulate(question, history)
    hits = retrieve(collection, model, q)
    return ask_groq(q, hits, history)


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])

    collection, model = get_retriever()
    hits = retrieve(collection, model, question)
    print(f"Retrieved {len(hits)} chunk(s):")
    for h in hits:
        print(f"  [{h['distance']:.3f}] {h['meta']['title']} - {h['meta']['url']}")

    api_key = load_api_key()
    if not api_key:
        print()
        print(
            "No Groq API key found (env GROQ_API_KEY or .streamlit/secrets.toml). "
            "Retrieval works; set the key to get LLM answers."
        )
        sys.exit(2)

    try:
        print(f"\nAnswer:\n{ask_groq(question, hits)}")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()