# Multi-Tool ReAct Agent (with Document Retrieval)

An LLM-powered agent that reasons about multi-step tasks and decides which tool to call — calculator, web search, Python code execution, or document retrieval — using the ReAct (Reason + Act) pattern. Built with Google's Gemini API.

## Why an agent, not just an LLM call

A single LLM call is reactive: ask a question, get an answer, done. It can't reliably do exact arithmetic, doesn't know anything after its training cutoff, and can't ground answers in a specific document set. An agent addresses this by looping: the LLM reasons about what it needs, calls a real tool to get it, observes the result, and decides whether it has enough information to answer or needs another step.

## Architecture

1. **Thought** — the LLM reasons about what to do next
2. **Action** — it specifies a tool call, e.g. `retrieve_documents("death penalty")`
3. **Observation** — the tool actually runs (outside the LLM), and its real output is fed back
4. **Repeat** until the LLM has enough information, then it outputs a **Final Answer**

## Tools

- **`calculator(expression)`** — restricted AST-based math evaluator (not `eval()`, which allows arbitrary code execution)
- **`web_search(query)`** — DuckDuckGo search (`ddgs` library, no API key required)
- **`execute_code(code)`** — Python execution in a restricted namespace (limited builtins, no file/import access)
- **`retrieve_documents(query)`** — hybrid retrieval + cross-encoder re-ranking over an indexed document set (built and tested against the Constitution of Nepal)

## The retrieval tool's debugging journey

Building a genuinely reliable `retrieve_documents` tool took three iterations, each revealing a different real limitation of retrieval techniques:

**1. Naive score blending (BM25 + embeddings, normalized and summed) — failed.**
Query: *"What does the constitution say about the death penalty?"* The correct article (16, "No law shall be made for capital punishment") was buried because an unrelated article (301) scored an inflated BM25 value from generic keyword overlap ("constitution," "provisions"), skewing the normalization.

**2. Reciprocal Rank Fusion (RRF) — also failed.**
RRF combines rankings instead of raw scores, which normally protects against exactly this kind of score-scale skew. But diagnosis showed Article 16 ranked #1 in embeddings yet **276th out of 285 in BM25** (since "capital punishment" shares almost no words with "death penalty"), while Article 301 ranked #1 in *both* signals. RRF assumes the two methods make independent errors that cancel out — but here both methods independently and correlated-ly preferred the wrong article, so fusion couldn't recover the right one.

**3. Cross-encoder re-ranking — succeeded.**
Instead of comparing separately-computed query/document embeddings, a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores the query and each candidate document *jointly*, letting it directly reason about whether "capital punishment" and "death penalty" refer to the same concept. Retrieving a wider candidate set (top 15 by embeddings) and re-ranking with the cross-encoder correctly surfaced Article 16 at rank 1.

**Key lesson:** hybrid search (naive or RRF) only helps when different retrieval methods make *different* mistakes that offset each other. When methods agree on the same wrong answer, only a technique that can reason about the query and document jointly (like a cross-encoder) can fix it.

One further, smaller finding: the cross-encoder's ranking is somewhat sensitive to query phrasing — a short query ("death penalty") ranked the correct article lower than the full natural-language question did, suggesting production use should prefer passing the agent's full question to the retriever rather than an extracted keyword phrase.

## Agent tool-selection test

Given the question *"What does the Nepal constitution say about the death penalty?"*, the agent correctly reasoned that this required constitutional document search and selected `retrieve_documents` (not `web_search`) — confirming the tool-selection logic works as intended for domain-specific questions.

A full multi-step trace (retrieval → synthesis) was limited by the Gemini free-tier daily quota (20 requests/day) during testing; the tool-selection step completed successfully before the quota was hit.

## Structure
```
src/
  tools.py   # calculator, web_search, execute_code, DocumentRetriever (hybrid + cross-encoder)
  agent.py   # ReAct loop and tool dispatch
notebooks/   # step-by-step exploration notebook (Colab), with markdown explanations
```

## Stack
Python, Google Gemini API, `sentence-transformers` (bi-encoder + cross-encoder), FAISS, `rank_bm25`, `ddgs`

## What's next
- Complete a full end-to-end multi-step trace once API quota resets
- Investigate the query-phrasing sensitivity found in cross-encoder re-ranking
- Add more document sources to retrieval, testing whether the agent correctly picks among multiple document sets
- Apply this pattern to a domain-specific extension (e.g. NEPSE company filings/news retrieval)