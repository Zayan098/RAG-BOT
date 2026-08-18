"""Streamlit chat UI for the book chatbot.

Usage:
    streamlit run app.py
Requires GROQ_API_KEY in .streamlit/secrets.toml (or env) for LLM answers;
retrieval alone still works without it.
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st

import query

st.set_page_config(
    page_title="AI Agent Factory - Book Chatbot",
    page_icon=":material/menu_book:",
    layout="wide",
)

st.sidebar.title("AI Agent Factory")
st.sidebar.caption("RAG chatbot grounded in the book's 81 pages")
st.sidebar.markdown(
    "- Retrieval: **all-MiniLM-L6-v2** + ChromaDB (cosine)\n"
    f"- Top-k: **{query.TOP_K}**, chunk filter: **{query.MAX_CHUNK_DISTANCE}**\n"
    f"- LLM: **{query.PRIMARY_MODEL}** (fallback {query.FALLBACK_MODEL})"
)
st.sidebar.markdown("[Read the book](https://agentfactory.panaversity.org/)")

API_KEY_PRESENT = bool(query.load_api_key())
if not API_KEY_PRESENT:
    st.warning(
        "No Groq API key found. Add it to `.streamlit/secrets.toml` "
        "(GROQ_API_KEY = \"...\") or set the GROQ_API_KEY environment variable, "
        "then restart the app."
    )


@st.cache_resource(show_spinner="Loading the book index (first run takes a moment)...")
def get_retriever():
    return query.get_retriever()


st.title("Ask the Book")
st.caption(
    "Book questions are answered from the book's own content; general questions "
    "are answered like a regular assistant. Sources are shown when book content "
    "was used."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("**Sources:**")
            for label, url in msg["sources"]:
                st.markdown(f"- [{label}]({url})")


def build_sources(hits: list[dict]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    sources: list[tuple[str, str]] = []
    for h in hits:
        url = h["meta"]["url"]
        if url in seen:
            continue
        seen.add(url)
        sources.append((h["meta"]["title"], url))
    return sources


def respond(question: str, history: list[dict]) -> tuple[str, list[tuple[str, str]]]:
    greeting = query.greeting_reply(question)
    if greeting:
        return greeting, []
    collection, model = get_retriever()
    q = query.reformulate(question, history)
    hits = query.retrieve(collection, model, q)
    if not API_KEY_PRESENT:
        return (
            "No Groq API key is set, so I can't generate answers. Add "
            "GROQ_API_KEY to `.streamlit/secrets.toml` and restart the app.",
            build_sources(hits),
        )
    try:
        answer = query.ask_groq(q, hits, history)
    except RuntimeError as e:
        return f"Sorry, I couldn't answer that: {e}", build_sources(hits)
    return answer, build_sources(hits)


if prompt := st.chat_input("Ask about the book..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the book..."):
            answer, sources = respond(prompt, st.session_state.messages)
        st.markdown(answer)
        if sources:
            st.markdown("**Sources:**")
            for label, url in sources:
                st.markdown(f"- [{label}]({url})")
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )