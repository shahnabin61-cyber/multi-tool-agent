# Multi-Tool ReAct Agent

An LLM-powered agent that reasons about multi-step tasks and decides which tool to call — calculator, web search, or Python code execution — using the ReAct (Reason + Act) pattern. Built with Google's Gemini API.

## Why an agent, not just an LLM call

A single LLM call is reactive: ask a question, get an answer, done. It can't reliably do exact arithmetic, doesn't know anything after its training cutoff, and can't run real computation. An agent addresses this by looping: the LLM reasons about what it needs, calls a real tool to get it, observes the result, and decides whether it has enough information to answer or needs another step.

## Architecture

1. **Thought** — the LLM reasons about what to do next
2. **Action** — it specifies a tool call, e.g. `web_search("population of Nepal")`
3. **Observation** — the tool actually runs (outside the LLM), and its real output is fed back
4. **Repeat** until the LLM has enough information, then it outputs a **Final Answer**

## Tools

- **`calculator(expression)`** — evaluates math expressions using a restricted AST-based evaluator (not Python's `eval()`, which would allow arbitrary code execution)
- **`web_search(query)`** — searches the web via DuckDuckGo (`ddgs` library, no API key required)
- **`execute_code(code)`** — runs Python code in a restricted namespace (limited builtins only, no file/import access)

Each tool was tested independently before wiring into the agent loop:
- Calculator: correctly evaluated `47 * 892`, `(15+3)/2`, `2**10`
- Web search: returned real, current results (e.g. Nepal's 2026 population data from Worldometer)
- Code execution: correctly ran list comprehensions and aggregation functions

## Test run: multi-step reasoning

Test question: *"What is 15% of the current population of Nepal?"* — this requires two sequential steps (search, then calculate), which a single LLM call cannot do reliably (LLMs frequently hallucinate current statistics and cannot reliably do arithmetic without hallucination).

**Result (Step 1):** the agent correctly reasoned `"I need to find the current population of Nepal first"`, called `web_search("current population of Nepal 2024")`, and received real data (~29.6 million from Worldometer, September 2026). This confirms the core ReAct loop — reasoning followed by a correct tool selection — works as designed.

The full run (search → calculate → final answer) was cut short by hitting the Gemini free-tier daily quota (20 requests/day) during testing, shared across this and a concurrent RAG project built the same day. Given each of the three tools was independently verified and Step 1 of the full agent loop executed correctly, the architecture is sound; a full end-to-end trace is straightforward to capture with a fresh quota.

## Structure
```
src/
  tools.py   # calculator, web_search, execute_code
  agent.py   # ReAct loop and tool dispatch
notebooks/   # step-by-step exploration notebook (Colab), with markdown explanations
```

## Stack
Python, Google Gemini API, `ddgs`

## What's next
- Complete a full end-to-end multi-step trace once API quota resets
- Add more tools (e.g. a weather API, a file reader) to test tool-selection accuracy as options grow
- Add retry/error-handling for malformed agent actions (currently returns an error string but doesn't let the agent self-correct)
- Apply this agent pattern to NEPSE data (fetch stock data + compute stats + reason about trends) as a domain-specific extension