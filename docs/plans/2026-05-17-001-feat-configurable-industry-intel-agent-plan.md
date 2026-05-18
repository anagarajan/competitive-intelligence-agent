---
title: "feat: Configurable industry competitive intelligence agent"
type: feat
status: active
date: 2026-05-17
---

# feat: Configurable Industry Competitive Intelligence Agent

## Summary

Build the full competitive intelligence multi-agent system from the spec in `competitive_intel_agent_prompt.md` — Researcher → Analyst → Writer on a LangGraph `StateGraph` — with industry configurability baked in from the start. Instead of hardcoded GenAI companies, the user supplies `--industry`, `--niche`, and `--companies` at runtime via CLI. Agent system prompts are parameterized with that context so search queries, significance scoring, and briefing prose adapt to any domain. The system supports both Claude (Anthropic) and OpenAI as LLM backends via a `--provider` flag; all three agents route through a thin provider abstraction so switching costs nothing.

---

## Problem Frame

The spec defines a system tailored to GenAI companies. The hardcoded company list and GenAI-specific agent framing prevent reuse across industries. The user needs to monitor a custom domain of their choosing, specified at runtime without touching source code.

---

## Requirements

- R1. User supplies `--companies` (required, space-separated list) via CLI
- R2. User supplies `--industry` (required) and `--niche` (optional) via CLI
- R3. Researcher agent's Tavily search queries incorporate `industry` and `niche` context
- R4. Analyst agent scores significance relative to the provided industry context
- R5. Writer agent produces briefings framed for the provided industry
- R6. System runs end-to-end: raw news → scored findings → markdown briefing saved to `output/briefing_{date}.md`
- R7. Errors (Tavily failures, LLM call failures) are logged to `state["errors"]` and printed after the briefing; they never crash the run
- R8. If all findings are dropped (all `confidence: "low"`), Writer explicitly states no verifiable developments found — does not hallucinate content
- R9. User can select the LLM backend via `--provider claude` (default) or `--provider openai`; an optional `--model` flag overrides the default model name for that provider

---

## Scope Boundaries

- Auto-discovery of companies from industry name is explicitly out of scope — `--companies` is always required
- Slack or email delivery of briefings is out of scope
- "Last week's briefing" comparison is out of scope
- Fact-checker 4th agent is out of scope
- Persistent storage or database of briefing history is out of scope

### Deferred to Follow-Up Work

- Company auto-discovery from `--industry` alone: future iteration
- 4th Fact Checker agent: future iteration (spec stretch goal)
- Slack/email delivery: future iteration
- `--last-briefing` comparison input: future iteration

---

## Context & Research

### Relevant Code and Patterns

- `competitive_intel_agent_prompt.md` — authoritative source for agent behavior, state schema, output format, and error handling contract
- No existing implementation; this is a greenfield build

### External References

- LangGraph `StateGraph` with `TypedDict` state — linear flow with `START → node → END`, each node is `(state) -> state`
- `langchain_community.tools.tavily_search.TavilySearchResults(max_results=5)` — bound to Researcher only
- `langchain_anthropic.ChatAnthropic` — Claude backend; model default `claude-sonnet-4-20250514`
- `langchain_openai.ChatOpenAI` — OpenAI backend; model default `gpt-4o`
- Both `ChatAnthropic` and `ChatOpenAI` implement the same LangChain `BaseChatModel` interface — swapping providers requires no changes to agent logic
- Python `argparse` (stdlib) — avoids adding `click` as a dependency

---

## Key Technical Decisions

- **argparse over click**: stdlib keeps dependencies lean; the CLI surface (3–4 flags) doesn't need click's ergonomics.
- **`Config` dataclass**: A single `@dataclass` carrying `industry`, `niche`, `companies`, (optional) `search_focus` is constructed from CLI args and passed into the graph. Explicit and testable vs environment-level globals.
- **Parameterized prompts via f-strings**: Agent system prompts are constructed as f-strings accepting `config.industry` and `config.niche`. Simpler than a template engine for this use case and easy to test.
- **Industry context in Tavily queries**: Researcher prepends `industry + niche` to each per-company search query so results stay domain-relevant even for generic company names (e.g. "Linear" in a B2B SaaS niche vs. the algebra software).
- **Factory functions for node binding**: Each agent node is created by a factory (`make_researcher_node(config)`) that closes over `Config`. This lets the pure `StateGraph` definition remain clean while injecting config.
- **State schema from spec**: Use the spec's `TypedDict` exactly — `companies`, `researcher_output`, `analyst_output`, `final_briefing`, `errors`. No additions without a clear need.
- **Temperatures from spec**: Researcher 0.1, Analyst 0.2, Writer 0.4.
- **LLM provider abstraction via LangChain's unified interface**: Both `ChatAnthropic` and `ChatOpenAI` implement `BaseChatModel`; a `get_llm(config, temperature)` factory in `competitive_intel/llm.py` returns the right instance based on `config.provider`. Agent nodes call the factory — they never import a provider class directly. This makes providers swappable with zero agent code changes.
- **Default models per provider**: Claude default is `claude-sonnet-4-20250514` (from spec); OpenAI default is `gpt-4o`. Both are overridable via `config.model`.
- **Separate API key env vars per provider**: `ANTHROPIC_API_KEY` for Claude, `OPENAI_API_KEY` for OpenAI. Only the active provider's key is required at runtime.

---

## Open Questions

### Resolved During Planning

- **CLI library**: argparse (stdlib) — surface is simple, no extra dep.
- **`--companies` format**: `nargs='+'` so users write `--companies Stripe Plaid Adyen` (space-separated, no quotes needed for single words).
- **Config propagation into LangGraph nodes**: Factory function / closure pattern (`make_*_node(config)`) — compatible with any LangGraph version.
- **`--provider` default**: `claude` — preserves spec-default behavior when the flag is omitted.

### Deferred to Implementation

- Exact Tavily query templates per category — implementer should iterate with real search results for the target industry
- Whether `with_structured_output()` or explicit JSON-mode prompting works better for Researcher/Analyst structured output — test at implementation time
- LangGraph version-specific node registration API — verify against the installed version before coding graph.py

---

## Output Structure

```
competitive_intel/
├── __init__.py
├── config.py          # Config dataclass (industry, companies, provider, model, …)
├── llm.py             # get_llm(config, temperature) — provider abstraction
├── main.py            # CLI entry point + run_briefing()
├── graph.py           # StateGraph + IntelState TypedDict
├── agents/
│   ├── __init__.py
│   ├── researcher.py
│   ├── analyst.py
│   └── writer.py
├── tools/
│   ├── __init__.py
│   └── search.py      # Tavily tool factory
└── output/            # runtime-generated briefings (gitignored)
requirements.txt
.env.example
.gitignore
tests/
├── __init__.py
├── test_config.py
├── test_llm.py
├── test_researcher.py
├── test_analyst.py
├── test_writer.py
└── test_graph.py
```

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
CLI (argparse)
  --industry "B2B SaaS"
  --niche "product-led growth tools"
  --companies Notion Linear Figma Loom Retool
  --provider openai          # or "claude" (default)
  --model gpt-4o             # optional override
         │
         ▼
    Config(industry, niche, companies, provider, model)
         │
         ▼
   run_briefing(config)  ← main.py
         │
   ┌─────▼─────────────────────────────────────────────┐
   │              LangGraph StateGraph                  │
   │                                                    │
   │  IntelState:                                       │
   │    companies: list[str]                            │
   │    researcher_output: list[CompanyFindings]        │
   │    analyst_output: AnalystOutput | None            │
   │    final_briefing: str                             │
   │    errors: list[str]                               │
   │                                                    │
   │  START                                             │
   │    → make_researcher_node(config)  ← get_llm()    │
   │    → make_analyst_node(config)     ← get_llm()    │
   │    → make_writer_node(config)      ← get_llm()    │
   │  → END                                             │
   └────────────────────────────────────────────────────┘
         │
         ├── print final_briefing to stdout
         ├── save output/briefing_{YYYY-MM-DD}.md
         └── print errors (if any)
```

Researcher search query shape (directional — implementer refines with real results):
```
"{company} {industry} {niche} {category} news last 7 days"
```

---

## Implementation Units

### U1. Project Scaffold and Config Schema

**Goal:** Create the full directory structure, `requirements.txt`, `.env.example`, `.gitignore`, and the `Config` dataclass that flows through the entire system.

**Requirements:** R1, R2

**Dependencies:** None

**Files:**
- Create: `competitive_intel/__init__.py`
- Create: `competitive_intel/config.py`
- Create: `competitive_intel/agents/__init__.py`
- Create: `competitive_intel/tools/__init__.py`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Approach:**
- `Config` is a `@dataclass` with: `industry: str`, `companies: list[str]`, `niche: str = ""`, `search_focus: list[str]` (default empty list), `provider: str = "claude"`, `model: str | None = None`
- `model` defaults to `None`; `llm.py` applies the per-provider default when it is `None`
- `requirements.txt` pins: `langgraph`, `langchain`, `langchain-anthropic`, `langchain-openai`, `langchain-community`, `tavily-python`, `anthropic`, `openai`, `python-dotenv`
- `.env.example` shows `ANTHROPIC_API_KEY=`, `OPENAI_API_KEY=`, and `TAVILY_API_KEY=` with a comment that only the active provider's key is required
- `.gitignore` covers `output/`, `.env`, `__pycache__/`, `*.pyc`, `.pytest_cache/`

**Test scenarios:**
- Happy path: `Config(industry="FinTech", companies=["Stripe", "Plaid"])` constructs without error; `provider` defaults to `"claude"`, `model` defaults to `None`
- Happy path: `Config(industry="FinTech", companies=["Stripe"], provider="openai")` constructs without error
- Edge case: `Config` with `niche=""` (default) is valid and produces empty string, not `None`
- Edge case: `Config` with single-item `companies=["Stripe"]` — field is `list[str]`, not bare string

**Verification:**
- `Config` instantiates and its fields are the expected types
- `requirements.txt` installs without conflicts in a fresh venv (`pip install -r requirements.txt`)

---

### U2. CLI Entry Point

**Goal:** Implement `main.py` with argparse CLI that parses `--industry`, `--companies`, `--niche`, constructs a `Config`, and calls `run_briefing(config)` which runs the graph, saves the briefing, and prints errors.

**Requirements:** R1, R2, R6, R7

**Dependencies:** U1

**Files:**
- Create: `competitive_intel/main.py`
- Modify: `tests/test_config.py` (add CLI parsing tests)

**Approach:**
- `--industry`: required string
- `--companies`: `nargs='+'`, required — produces `list[str]`
- `--niche`: optional string, default `""`
- `--provider`: optional, choices `["claude", "openai"]`, default `"claude"`
- `--model`: optional string, default `None` — overrides the provider's default model name
- Load `.env` via `python-dotenv` at module top, before any API client initialization
- `run_briefing(config)` initializes state, compiles and runs the graph, prints briefing to stdout, saves to `output/briefing_{date}.md` (creating `output/` if needed), then prints any errors from `state["errors"]`

**Test scenarios:**
- Happy path: `parse_args(["--industry", "FinTech", "--companies", "Stripe", "Plaid"])` → `Config(industry="FinTech", companies=["Stripe","Plaid"], niche="", provider="claude", model=None)`
- Happy path: `parse_args([..., "--provider", "openai"])` → `Config.provider == "openai"`
- Happy path: `parse_args([..., "--provider", "openai", "--model", "gpt-4-turbo"])` → `Config.model == "gpt-4-turbo"`
- Error path: Missing `--industry` → argparse exits with a non-zero code and an error message
- Error path: Missing `--companies` → argparse exits with a non-zero code and an error message
- Error path: `--provider gemini` (invalid choice) → argparse exits with error listing valid choices
- Edge case: Single `--companies Stripe` → `Config.companies == ["Stripe"]` (list, not string)
- Edge case: `output/` directory does not exist → `run_briefing` creates it before writing the file

**Verification:**
- `python -m competitive_intel.main --industry "FinTech" --companies Stripe Plaid` runs without import errors
- `output/briefing_YYYY-MM-DD.md` is created after a successful run

---

### U3. Tavily Search Tool and Researcher Agent

**Goal:** Implement `tools/search.py` (Tavily tool factory) and `agents/researcher.py` (Agent 1). The Researcher issues per-company, industry-aware Tavily searches and returns structured findings with `confidence` flags as defined in the spec.

**Requirements:** R3, R7

**Dependencies:** U1

**Files:**
- Create: `competitive_intel/tools/search.py`
- Create: `competitive_intel/agents/researcher.py`
- Create: `tests/test_researcher.py`

**Approach:**
- `search.py` exposes `get_search_tool()` returning `TavilySearchResults(max_results=5)`, instantiated lazily after env vars are loaded
- Researcher node factory: `make_researcher_node(config) -> Callable[[IntelState], IntelState]`
- System prompt parameterized with `config.industry` and `config.niche`; instructs the model to search for the 6 spec categories: `product_launch`, `funding`, `partnership`, `research`, `hiring`, `regulatory`
- Search query per company: `"{company} {config.industry} {config.niche} {category} news last 7 days"` (implementer refines based on real results)
- If Tavily raises an exception for a company → append to `state["errors"]`, continue with remaining companies
- If Tavily returns 0 results for a finding → mark `confidence: "low"` on that finding
- Structured output per company: `{"company": str, "findings": [{"headline", "source_url", "date", "summary", "category", "confidence"}]}`
- LLM obtained from `get_llm(config, temperature=0.1)` — never hardcode `ChatAnthropic`

**Test scenarios:**
- Happy path: Researcher called with `Config(industry="FinTech", companies=["Stripe"])` and mocked Tavily returning 3 results → `researcher_output` contains one `CompanyFindings` entry with 3 findings, each with all required fields
- Error path: Tavily raises an exception for "Stripe" → error appended to `state["errors"]`; if other companies exist, they are still processed
- Edge case: Tavily returns 0 results for one company → that company's findings are marked `confidence: "low"`; run continues
- Integration: Researcher node takes `IntelState` dict, returns updated `IntelState` with `researcher_output` populated and no uncaught exceptions

**Verification:**
- Running the Researcher node in isolation with a single mocked company returns a properly structured `researcher_output`
- Tavily exceptions populate `state["errors"]` without crashing the node

---

### U4. Analyst Agent

**Goal:** Implement `agents/analyst.py` (Agent 2). Receives Researcher output, silently drops all `confidence: "low"` findings before sending to the LLM, assigns `HIGH/MEDIUM/LOW` significance relative to the configured industry, identifies cross-company trends, and produces a Watch List.

**Requirements:** R4, R7, R8

**Dependencies:** U3

**Files:**
- Create: `competitive_intel/agents/analyst.py`
- Create: `tests/test_analyst.py`

**Approach:**
- Pure LLM call, no tools — `get_llm(config, temperature=0.2)` from `competitive_intel/llm.py`
- Low-confidence findings are filtered out in Python *before* the LLM prompt is constructed — the LLM never sees them
- System prompt parameterized with `config.industry` and `config.niche`; instructs model to score significance relative to that industry
- Structured output: scored findings per company + `cross_company_trends: list[str]` + `watch_list: list[str]` (2–3 items) + `zero_valid_findings: bool`
- If zero valid findings remain after filtering → set `zero_valid_findings: true` in output; Writer consumes this flag
- Retry once on LLM call failure; on second failure, log to `state["errors"]` and return a partial `analyst_output` with `zero_valid_findings: true`
- Cross-company trends: themes appearing in ≥ 2 companies (as specced)

**Test scenarios:**
- Happy path: 5 findings across 2 companies (all `confidence: "high"`) → `analyst_output` has scored findings with `HIGH/MEDIUM/LOW`, a non-empty `cross_company_trends` list, and a `watch_list` of 2–3 items
- Edge case: All input findings are `confidence: "low"` → Analyst LLM prompt receives empty findings list → `zero_valid_findings: true` in output
- Edge case: Only 1 company has valid findings → `cross_company_trends` is empty list, not an error, not hallucinated
- Error path: LLM call fails twice → error logged to `state["errors"]`; `analyst_output` returned with `zero_valid_findings: true`
- Integration: Analyst node takes `IntelState` with `researcher_output` set, returns state with `analyst_output` populated

**Verification:**
- No `confidence: "low"` findings appear in any LLM prompt sent by the Analyst node (verifiable via prompt capture in tests)
- `analyst_output` JSON has all required fields and parses cleanly

---

### U5. Writer Agent

**Goal:** Implement `agents/writer.py` (Agent 3). Receives Analyst output and produces the formatted markdown briefing. Includes only HIGH and MEDIUM findings. Handles zero-findings state gracefully.

**Requirements:** R5, R8

**Dependencies:** U4

**Files:**
- Create: `competitive_intel/agents/writer.py`
- Create: `tests/test_writer.py`

**Approach:**
- Pure LLM call, no tools — `get_llm(config, temperature=0.4)` from `competitive_intel/llm.py`
- System prompt parameterized with `config.industry`; provides the exact briefing format from the spec and instructions for each section
- Writer instruction: only HIGH and MEDIUM significance findings; for companies with no HIGH/MEDIUM findings, include one line: `"{Company}: No significant developments this week."`
- If `analyst_output.zero_valid_findings` is `true` → Writer is instructed to state clearly that no verifiable developments were found; no fabricated content
- Output: plain markdown string in the exact format from the spec (EXECUTIVE SUMMARY, COMPANY SNAPSHOTS, CROSS-CUTTING TRENDS, WATCH LIST)

**Test scenarios:**
- Happy path: Analyst output with 3 HIGH, 2 MEDIUM, 1 LOW finding across 2 companies → briefing includes 5 findings, all 4 required sections, excludes the LOW finding
- Edge case: Company in `companies` list has no HIGH or MEDIUM findings → briefing includes `"{Company}: No significant developments this week."` for that company
- Edge case: `zero_valid_findings: true` in Analyst output → briefing explicitly states no verifiable developments found; none of the standard finding lines appear
- Integration: Writer node takes `IntelState` with `analyst_output` populated, returns state with `final_briefing` as a non-empty markdown string

**Verification:**
- Briefing output contains all 4 required sections from the spec format
- No LOW significance findings appear in the briefing
- `zero_valid_findings` path produces a coherent briefing that does not hallucinate company developments

---

### U6. LangGraph State Graph and Wiring

**Goal:** Implement `graph.py` defining `IntelState` TypedDict, the `StateGraph`, node registration via factory functions, and `run_briefing(config)`. Wire all three agents into the linear `START → researcher → analyst → writer → END` flow.

**Requirements:** R6, R7

**Dependencies:** U3, U4, U5

**Files:**
- Create: `competitive_intel/graph.py`
- Create: `tests/test_graph.py`

**Approach:**
- `IntelState(TypedDict)` fields: `companies: list[str]`, `researcher_output`, `analyst_output`, `final_briefing: str`, `errors: list[str]`
- Each node is registered via factory closure: `graph.add_node("researcher", make_researcher_node(config))`
- `StateGraph` uses `IntelState`; edges connect `START → researcher → analyst → writer → END`
- `run_briefing(config)` initializes state as `{"companies": config.companies, "researcher_output": [], "analyst_output": None, "final_briefing": "", "errors": []}`, compiles the graph, invokes it, then:
  1. Prints `final_briefing` to stdout
  2. Creates `output/` if absent, writes `output/briefing_{YYYY-MM-DD}.md`
  3. Prints all entries in `errors` (if any)

**Test scenarios:**
- Happy path: Full graph run with mocked Researcher, Analyst, Writer node functions → state flows through all three nodes; `final_briefing` is non-empty; `errors` is empty
- Error path: Researcher node logs an error to `state["errors"]` for one company → graph completes; `state["errors"]` is non-empty and printed after briefing; `final_briefing` is still produced
- Integration: End-to-end run with real Tavily and LLM calls (requires `.env`) produces a non-empty `output/briefing_{date}.md` and exits cleanly

**Verification:**
- `run_briefing(Config(industry="FinTech", companies=["Stripe", "Plaid"]))` completes without uncaught exceptions (with valid API keys)
- `output/briefing_YYYY-MM-DD.md` file exists and is non-empty after a successful run
- Any accumulated errors are printed to stdout after the briefing, not before

---

### U7. LLM Provider Abstraction

**Goal:** Implement `competitive_intel/llm.py` — a `get_llm(config, temperature)` factory that returns the correct LangChain chat model instance (`ChatAnthropic` or `ChatOpenAI`) based on `config.provider` and `config.model`. All three agent nodes call this factory; no agent imports a provider class directly.

**Requirements:** R9

**Dependencies:** U1

**Files:**
- Create: `competitive_intel/llm.py`
- Create: `tests/test_llm.py`

**Approach:**
- `get_llm(config, temperature: float)` returns a `BaseChatModel` instance
- When `config.provider == "claude"`: returns `ChatAnthropic(model=config.model or "claude-sonnet-4-20250514", temperature=temperature)`
- When `config.provider == "openai"`: returns `ChatOpenAI(model=config.model or "gpt-4o", temperature=temperature)`
- Unknown `config.provider` raises `ValueError` with a clear message listing valid choices — fail fast at startup, not mid-run
- No caching or singleton; agents call `get_llm(config, temperature)` at node construction time (inside each factory closure), not on every invocation

**Test scenarios:**
- Happy path: `get_llm(Config(provider="claude", ...), 0.1)` returns an instance of `ChatAnthropic` with temperature 0.1 and model `claude-sonnet-4-20250514`
- Happy path: `get_llm(Config(provider="openai", ...), 0.2)` returns an instance of `ChatOpenAI` with temperature 0.2 and model `gpt-4o`
- Happy path: `get_llm(Config(provider="openai", model="gpt-4-turbo", ...), 0.4)` returns `ChatOpenAI` with `model="gpt-4-turbo"`
- Error path: `get_llm(Config(provider="gemini", ...), 0.1)` raises `ValueError` with message naming the invalid provider and listing valid choices
- Edge case: `config.model = None` with `provider="claude"` → factory applies `"claude-sonnet-4-20250514"` default without error

**Verification:**
- Calling `get_llm` with valid providers returns the expected LangChain class instances
- Agent files (`researcher.py`, `analyst.py`, `writer.py`) contain no direct imports of `ChatAnthropic` or `ChatOpenAI`

---

## System-Wide Impact

- **Interaction graph:** `Config` flows into all three agent nodes via factory closures; changes to `--industry` or `--niche` at the CLI propagate to Tavily query text, LLM system prompts, and briefing framing
- **Error propagation:** Errors accumulate in `state["errors"]` list throughout the graph; nodes append, never overwrite; `run_briefing` prints them at the end — the flow is never interrupted
- **State lifecycle risks:** LangGraph nodes return new state dicts (functional update pattern); no in-place mutation
- **API surface parity:** `run_briefing(config)` is directly importable for scripting and testing, not just CLI-reachable
- **Integration coverage:** The full Tavily → LLM → file chain requires either real API keys or a comprehensive mock chain; unit tests should mock both; one integration test should run with real keys (skipped in CI without them)
- **Unchanged invariants:** The spec's exact briefing output format (section names, prefixes `•`, `→`, `!`) and `IntelState` field names are preserved as defined

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Tavily returns off-topic results for non-AI companies (e.g., "Linear" pulled as math software) | Industry+niche context baked into every query; implementer iterates on query templates with real results before locking in |
| LLM structured output (Researcher/Analyst) fails to parse reliably | Use `with_structured_output()` or explicit JSON-mode prompt; retry once on parse failure; log on second failure |
| LangGraph API varies between versions | Pin all versions in `requirements.txt`; implementer verifies `StateGraph` node API against installed version |
| Briefing format drift when `zero_valid_findings` path is hit | Explicit Writer instruction + dedicated test scenario covers this path |
| OpenAI structured output behavior differs from Claude (e.g., JSON mode quirks, tool-calling format) | `with_structured_output()` is tested against both providers at implementation time; implementer notes any divergence and adjusts prompts per provider if needed |
| User supplies `--provider openai` but only has `ANTHROPIC_API_KEY` set | `get_llm` raises at node construction time (inside `run_briefing`) before the graph runs; error message names the missing key |

---

## Documentation / Operational Notes

- Update `CLAUDE.md` after implementation with the actual run command and any deviations from the specced file structure
- `output/` directory must be in `.gitignore` — generated files should not be committed
- `.env` must be in `.gitignore`; `.env.example` must be committed

---

## Sources & References

- **Specification:** `competitive_intel_agent_prompt.md`
- **Architecture reference:** `CLAUDE.md`
