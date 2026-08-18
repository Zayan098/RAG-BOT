"""Evaluation runner for the book chatbot.

Reads eval_questions.md, runs each question through the pipeline, and reports
mode correctness + answer hygiene. Use --llm to also call Gemini (costs quota);
without it, only retrieval/greeting checks run (no API cost).

Usage:
    python eval.py            # dry run: mode + hygiene only
    python eval.py --llm      # full: also generate + score answers
    python eval.py --llm --out results.md   # save answers for manual review
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import query

Q_RE = re.compile(r"^-\s*\[(\w+)\]\s*(.+)$")
LEAK_RE = re.compile(r"https?://|www\.|panaversity\.org|\.md\b|chunk|context|document", re.IGNORECASE)


def load_questions() -> list[tuple[str, str, str]]:
    path = Path(__file__).parent / "eval_questions.md"
    items: list[tuple[str, str, str]] = []
    category = "misc"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            category = line[3:].strip()
        m = Q_RE.match(line)
        if m:
            items.append((category, m.group(1), m.group(2).strip()))
    return items


def check_hygiene(text: str) -> list[str]:
    failures = []
    if not text or not text.strip():
        failures.append("empty answer")
    if LEAK_RE.search(text):
        failures.append("leak: URL/path/jargon present")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the eval question set.")
    parser.add_argument("--llm", action="store_true", help="Call Gemini for full answers")
    parser.add_argument("--out", type=str, default="", help="Write full answers to this file")
    args = parser.parse_args()

    items = load_questions()
    print(f"Loaded {len(items)} question(s) from eval_questions.md\n")

    retriever = query.get_retriever()
    collection, model = retriever
    out_lines: list[str] = []
    stats = {"book": 0, "general": 0, "greeting": 0, "followup": 0}
    passed = 0
    total = 0

    history: list[dict] = []
    for category, expected, question in items:
        if expected == "greeting":
            answer = query.greeting_reply(question) or ""
            mode = "greeting" if answer else "general"
        else:
            if expected == "followup":
                q = query.reformulate(question, history)
                hits = query.retrieve(collection, model, q)
            else:
                q = question
                hits = query.retrieve(collection, model, q)
            mode = "book" if hits else "general"
            if args.llm and query.load_api_key():
                try:
                    answer = query.ask_llm(q, hits, history)
                except RuntimeError as e:
                    answer = f"[ERROR] {e}"
            else:
                answer = "(dry run - no LLM call)"

        expected_mode = "book" if expected in ("book", "followup") else expected
        mode_ok = mode == expected_mode
        hygiene = [] if category == "greeting" else check_hygiene(answer)
        ok = mode_ok and not hygiene
        passed += 1 if ok else 0
        total += 1
        stats[expected] = stats.get(expected, 0) + 1

        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] [{category}] ({expected} -> {mode}) {question}")
        if not mode_ok:
            print(f"        mode mismatch: expected '{expected_mode}', got '{mode}'")
        for h in hygiene:
            print(f"        hygiene: {h}")
        if args.llm and answer.startswith("[ERROR]"):
            print(f"        {answer}")
        out_lines.append(f"### {question}  [{expected} -> {mode}] {flag}\n\n{answer}\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if category != "followup":
            history = history[-4:]

    print(f"\n{passed}/{total} passed")
    if args.out:
        Path(args.out).write_text("\n".join(out_lines), encoding="utf-8")
        print(f"Full answers written to {args.out}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()