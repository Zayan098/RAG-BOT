# Evaluation Questions

Each line is a question with an expected mode tag:

- `[book]` — question must be answered from book content (retrieval >= 1 chunk)
- `[general]` — unrelated to the book; must be answered as a general assistant (retrieval = 0 chunks)
- `[greeting]` — must get the canned greeting reply
- `[followup]` — depends on the previous question; tests conversation reformulation

Run with: `venv\Scripts\python.exe eval.py` (retrieval-only dry run, no API cost)
or `venv\Scripts\python.exe eval.py --llm` (also calls Groq for full answers).

## Concepts
- [book] What is a Digital FTE?
- [book] What is the agent loop?
- [book] What is a system of record, in plain words?
- [book] What is the two-layer model?
- [book] What are skills and connectors?

## Structure
- [book] What topics does the book cover?
- [book] What does the "Mode 2 - Manufacturing" path contain?
- [book] How is the book organized?
- [book] What does Phase 1 (Building Blocks) cover?

## Crash courses
- [book] What will I build in the Identic AI Chief of Staff crash course?
- [book] What is the context layer?
- [book] What is harness engineering?
- [book] How do I get paid for building AI agents?

## Out-of-book traps (must NOT be answered from the book)
- [general] What is the weather in Lahore?
- [general] What is the capital of France?
- [general] How do I make a simple Python function to reverse a string?
- [general] Tell me a joke.
- [book] What is a neural network? (book covers it in "What AI Actually Is" - book-grounded answer expected)

## Greetings
- [greeting] hello
- [greeting] salam
- [greeting] Hi!

## Follow-up pairs (conversation memory)
- [book] What is a Digital FTE?
- [followup] and how is it different from a regular chatbot?
- [book] What topics does the book cover?
- [followup] which ones are crash courses?
- [book] What is the agent loop?
- [followup] explain it again in simpler words

## Manual scoring guide (when reviewing --llm output)
For each answer score 0/1 on:
1. **Mode correctness**: book question got a book-grounded answer (no refusal, no generic fluff);
   general question got a normal helpful answer; greeting got the greeting.
2. **No leak**: answer contains no URLs, file paths, "chunk"/"context"/"document" jargon,
   or section-title link salad.
3. **Substance**: answer explains the concept in the user's terms - not a one-liner, not a
   list of headings. Follow-ups clearly refer back to the prior topic.
4. **Honesty**: if content was missing, the bot said so plainly instead of inventing facts.