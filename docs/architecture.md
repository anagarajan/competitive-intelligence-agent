# Competitive Intelligence Agent — A Complete Walkthrough

---

## The Problem We're Solving

Every week, things happen in your industry — competitors raise money, launch products, form partnerships, get sued. Staying on top of this manually means reading dozens of articles, synthesizing them, and writing a summary. That's 2-3 hours of work. This agent does it in 2-3 minutes.

The core challenge: **turning raw web search results into structured, actionable intelligence** — not just a list of links, but a ranked, filtered, opinionated brief.

---

## Why Multi-Agent Instead of One Big LLM Call?

The first instinct would be: give one LLM a prompt saying "search the web and write me a briefing." The problem is that **one agent trying to do everything becomes bad at everything**.

Think of it like a newsroom:

| Role | Responsibility | Why Separate |
|------|---------------|--------------|
| **Reporter** (Researcher) | Goes out, finds raw facts | Needs to be fast, literal, tool-using |
| **Editor** (Analyst) | Decides what matters | Needs to be analytical, comparative |
| **Writer** | Shapes the final story | Needs to be structured, consistent |

If you ask one person to simultaneously report, edit, and write — their judgment bleeds into their fact-finding, and their writing bleeds into their analysis. Separation of concerns produces better output.

---

## The Architecture: LangGraph

```
START → researcher_node → analyst_node → writer_node → END
```

**Why LangGraph over a simple function chain?**

You could just do:
```python
r = researcher(companies)
a = analyst(r)
w = writer(a)
```

We use LangGraph because:
1. **Shared state** — all agents read from and write to a single `IntelState` dict. No passing arguments around manually.
2. **Future extensibility** — adding a conditional branch (e.g. "if zero findings, skip writer") is one line in a graph, not a refactor.
3. **Observability** — LangGraph can trace every state transition, which helps debug when an agent produces bad output.

---

## Component 1: `config.py` — The Single Source of Truth

```python
@dataclass
class Config:
    industry: str
    companies: list[str]
    niche: str = ""
    provider: str = "claude"
    model: str | None = None
    search_focus: list[str] = field(default_factory=list)
```

**Why a dataclass?** It's a typed, immutable-by-default container. Compare to a plain dict — you'd have `config["industry"]` with no autocomplete, no type checking, and typos silently return `None`. The dataclass catches mistakes at definition time.

**Why does every agent receive Config instead of individual parameters?** The factory closure pattern:

```python
def make_researcher_node(config: Config) -> Callable[[dict], dict]:
    llm = get_llm(config, temperature=0.1)   # bound once at startup
    ...
    def researcher_node(state: dict) -> dict:
        ...  # config is captured in closure
    return researcher_node
```

The node is **created once** when the graph is built. The LLM client, search tool, and system prompt are initialized once and reused across all companies. You're not reconstructing expensive objects on every call.

---

## Component 2: `llm.py` — The Provider Abstraction

```python
def get_llm(config: Config, temperature: float):
    if config.provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=..., temperature=temperature)
    elif config.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=..., temperature=temperature)
```

**Why this instead of picking one provider?** Different providers have different strengths and costs. Claude is better at following strict formatting instructions (important for the Writer). OpenAI's GPT-4o is faster and cheaper for high-volume research. You might want Claude for production and OpenAI for testing.

**Why LangChain's wrappers?** Both `ChatAnthropic` and `ChatOpenAI` implement `BaseChatModel`, which means every agent calls `llm.invoke(messages)` without knowing or caring which provider is underneath. Swap providers with one flag — no agent code changes.

**Temperature choices are deliberate:**
- Researcher: `0.1` — low creativity, we want literal extraction of facts
- Analyst: `0.2` — slightly more judgment, but still grounded
- Writer: `0.4` — enough creativity for readable prose, not so much it hallucinates

---

## Component 3: `tools/search.py` — Tavily

```python
def get_search_tool() -> TavilySearch:
    return TavilySearch(max_results=5, topic="news")
```

**Why Tavily instead of Google Search API or DuckDuckGo?**

- Google's Custom Search API costs $5/1000 queries and requires complex setup
- DuckDuckGo's unofficial API gets rate-limited aggressively
- Tavily is purpose-built for LLM agents — it returns clean `{url, content, title}` dicts, not raw HTML that you then have to parse. It's the standard tool in the LangChain ecosystem for exactly this use case.

**Why `topic="news"`?** Without this, Tavily searches across all content including Wikipedia pages, product docs, and blog posts from 2019. The `news` topic restricts to recent news articles — exactly what we want for a weekly brief.

---

## Component 4: `agents/researcher.py` — The Data Collector

This is the most complex agent. It does five things in order:

### Step 1: Canonicalize company names

```python
canonical = _canonicalize_companies(llm, companies, config.industry)
```

The user types `"sentinal"` on the CLI. Tavily searches for `"sentinal"` and finds nothing because SentinelOne spells it differently. Before searching, we ask the LLM: *"What's the official name of these companies in this industry?"* The LLM maps `sentinal → SentinelOne`, `defender → Microsoft Defender`.

**Why use an LLM for this instead of a lookup table?** A lookup table would need to be maintained manually and would break for any company it doesn't know. The LLM knows company names across all industries and handles abbreviations, typos, and aliases.

### Step 2: Run two searches per company

```python
queries = [
    f"{canonical_name} {config.industry}{niche_part} news developments 2026",
    f"{canonical_name} {config.industry}{niche_part} funding acquisition partnership product launch 2026",
]
```

One search gives you whatever Tavily's algorithm decides is most relevant. Two searches with different terms casts a wider net — the first catches news coverage, the second catches business events (deals, launches, funding) that news articles might not surface with the same keywords.

### Step 3: Deduplicate by URL

```python
if url not in seen_urls:
    seen_urls.add(url)
    aggregated_results.append(r)
```

Both searches often return the same article. Without deduplication, the LLM would read it twice, potentially extracting the same finding twice and inflating the briefing.

### Step 4: Extract only niche-relevant findings

The system prompt explicitly says:

> *"Extract ONLY findings that are directly relevant to the {industry} industry. Ignore general company news unless it has a clear and specific impact on {industry}."*

Without this, searching "Google Ads" returns results about Gemini AI, Warby Parker glasses, and lawsuits — none of which are relevant to advertising. The LLM acts as a relevance filter, not just an extractor.

### Step 5: Confidence tagging

Each finding gets `"confidence": "high"` or `"low"`. Low means the source was vague, the claim was unverified, or the search returned no results. This is a signal to the next agent.

---

## Component 5: `agents/analyst.py` — The Filter and Scorer

```python
valid_findings = [
    item for item in researcher_output
    if any(f.get("confidence") != "low" for f in item["findings"])
]
```

**The most important design decision in the entire system happens here — in Python, not in the LLM prompt.**

Low-confidence findings are dropped *before* the LLM prompt is even constructed. Why? Because if you put garbage in an LLM prompt, the LLM will reason about the garbage. It might say "well, this source is vague but here's what I infer..." and invent analysis. By filtering in code first, the LLM never sees unverified data.

The analyst then:
- Assigns `HIGH / MEDIUM / LOW` significance based on competitive impact
- Identifies cross-company trends (themes appearing in 2+ companies)
- Produces a 2-3 item Watch List

**Why `zero_valid_findings: bool` in the output?** When everything is filtered out, the analyst returns this flag. The Writer checks it and explicitly states "no verifiable developments found" rather than fabricating content. This is the honest failure mode.

---

## Component 6: `agents/writer.py` — The Formatter

The Writer has the most constrained prompt in the system:

```
Write in this EXACT format — do not deviate from the structure, symbols, or section names:

WEEKLY COMPETITIVE INTELLIGENCE BRIEF
...
• finding [HIGH]
→ trend
! watch item
```

**Why such a rigid format?** Because the output is a product, not a conversation. If you let the LLM free-form it, you get different structures every week — sometimes bullet points, sometimes paragraphs, sometimes tables. A consistent format means readers know exactly where to look.

**Why only HIGH and MEDIUM findings?** An executive brief is not a news feed. If everything is reported, nothing stands out. LOW significance items are filtered here so the reader's attention is focused on what actually matters.

**Temperature 0.4 instead of 0.1?** At very low temperatures, prose becomes robotic and repetitive. We want readable sentences, not "Development identified: funding event occurred at company." Slightly higher temperature produces more natural writing while still following the format.

---

## Component 7: `graph.py` — The Wiring

```python
graph.add_edge(START, "researcher")
graph.add_edge("researcher", "analyst")
graph.add_edge("analyst", "writer")
graph.add_edge("writer", END)
```

This is intentionally simple. The graph is just wiring — it has no business logic. Each node is a pure function `(state: dict) -> dict` that reads what it needs, adds its output, and passes everything forward.

**Why does each node return `{**state, "new_key": value}` instead of just `{"new_key": value}`?** LangGraph merges return values into state, but being explicit with `**state` ensures the full state is preserved even if LangGraph's merging behavior changes. It also makes the code readable in isolation — you can see a node is passing everything through.

---

## Component 8: `main.py` — The CLI

```python
parser.add_argument("--industry", required=True)
parser.add_argument("--companies", nargs="+", required=True)
parser.add_argument("--niche", default="")
parser.add_argument("--provider", choices=["claude", "openai"], default="claude")
```

**Why argparse instead of click or typer?** Zero dependencies. Argparse is in Python's standard library. For a tool with 5 flags, the ergonomics difference is irrelevant and avoiding a dependency is always better.

**Why save to `output/briefing_{date}.md`?** So runs are never overwritten. If you run twice on a Monday and the second run has errors, you still have the first. It also creates a natural archive of weekly briefs without any database.

---

## The Error Handling Philosophy

Errors are collected but **never fatal**. If Tavily fails for CrowdStrike, the agent logs the error, continues to SentinelOne, and reports at the end:

```
--- ERRORS ---
• Tavily search failed for 'crowdstrike': connection timeout
```

Why? Because if one company's search fails, you still want the briefing for the other four. A fatal exception would give you nothing. A collected error gives you partial output plus a clear signal of what failed.

---

## How Everything Connects: A Full Data Flow

```
CLI args
  └─→ Config dataclass
        └─→ build_graph(config)
              └─→ make_researcher_node(config)  ← creates LLM + search tool once
              └─→ make_analyst_node(config)     ← creates LLM once
              └─→ make_writer_node(config)      ← creates LLM once
                    └─→ graph.invoke(initial_state)
                          ├─→ researcher: canonicalize → search × 2 → dedupe → extract
                          │     └─→ state["researcher_output"] = [{company, findings[]}]
                          ├─→ analyst: filter low-conf → score → trends → watch list
                          │     └─→ state["analyst_output"] = {scored_findings, trends, ...}
                          └─→ writer: format → HIGH/MEDIUM only → fixed structure
                                └─→ state["final_briefing"] = "WEEKLY COMPETITIVE..."
  └─→ print briefing
  └─→ save output/briefing_YYYY-MM-DD.md
  └─→ print errors (if any)
```

---

## The Key Insight

The whole system is designed around one principle: **each component does exactly one thing and does it well**. The Researcher doesn't analyze. The Analyst doesn't search. The Writer doesn't decide what's significant. Each agent's prompt is tightly scoped, which makes failures easy to diagnose and improvements easy to isolate.

When the briefing is wrong, you know exactly where to look: irrelevant findings → fix the Researcher prompt. Wrong significance scores → fix the Analyst prompt. Bad formatting → fix the Writer prompt.
