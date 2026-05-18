# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-agent competitive intelligence system that monitors GenAI companies and produces a weekly briefing. Built with LangGraph + Claude API + Tavily Search. The full specification is in `competitive_intel_agent_prompt.md`.

## Target File Structure

```
competitive_intel/
├── main.py              # entry point — run_briefing(companies: list[str])
├── graph.py             # LangGraph StateGraph definition
├── agents/
│   ├── researcher.py    # Agent 1: Tavily search, structured findings per company
│   ├── analyst.py       # Agent 2: scores significance, drops low-confidence, finds trends
│   └── writer.py        # Agent 3: produces markdown briefing in fixed format
├── tools/
│   └── search.py        # Tavily tool config (TavilySearchResults, max_results=5)
└── output/              # briefing_{YYYY-MM-DD}.md files saved here
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add ANTHROPIC_API_KEY and TAVILY_API_KEY
```

Run a briefing:
```bash
python -m competitive_intel.main
# or with custom companies:
python -c "from competitive_intel.main import run_briefing; run_briefing(['OpenAI', 'Anthropic'])"
```

## Architecture

### LangGraph Flow

```
START → researcher_node → analyst_node → writer_node → END
```

Single linear `StateGraph`. Each node is a plain function `(state) -> state`.

### Shared State Schema (`TypedDict`)

```python
{
  "companies": list[str],
  "researcher_output": list[CompanyFindings],   # raw findings per company
  "analyst_output": AnalystOutput,              # scored findings + trends + watch list
  "final_briefing": str,                        # finished markdown
  "errors": list[str]                           # non-fatal failures; printed at end
}
```

### Agent Responsibilities

**Researcher** — only agent with tool access. Calls Tavily per company, returns structured findings with `category` and `confidence` fields. Low-confidence findings are flagged but passed through.

**Analyst** — pure LLM call (no tools). Drops `confidence: "low"` findings, assigns `HIGH/MEDIUM/LOW` significance, identifies cross-company trends (≥2 companies), produces 2-3 item Watch List.

**Writer** — pure LLM call. Only writes HIGH/MEDIUM findings. Produces briefing in the exact format defined in the spec (see `competitive_intel_agent_prompt.md`).

### Model Config

All agents use `claude-sonnet-4-20250514`. Temperatures: Researcher `0.1`, Analyst `0.2`, Writer `0.4`.

Tavily bound to Researcher only via LangChain tool binding.

### Error Handling Contract

- Tavily failure for a company → log to `state["errors"]`, continue with remaining companies
- LLM call failure → retry once, then log and skip that agent step
- Zero valid findings after Analyst filtering → Writer explicitly states this rather than fabricating content
- Errors surface at the end of `run_briefing()`, never silently swallowed

## Build Sequence

Follow this order to iterate fast:
1. Researcher for a single company — confirm Tavily returns real results
2. Confirm structured output parses into state schema
3. Analyst with hardcoded Researcher output
4. Writer with hardcoded Analyst output
5. Wire into LangGraph graph
6. End-to-end with all 5 default companies
7. Add error handling layer

Default test companies: `["OpenAI", "Anthropic", "Google DeepMind", "Mistral", "Cohere"]`
