# Competitive Intelligence Agent — Build Prompt

## What you're building

A multi-agent system that automatically researches, analyzes, and reports on the GenAI competitive landscape. Three agents work in sequence: a Researcher that finds raw news and signals, an Analyst that synthesizes patterns and scores significance, and a Writer that produces a clean weekly briefing.

**Stack:** LangGraph + Claude API (claude-sonnet-4-20250514) + Tavily Search API + Markdown output

---

## Prompt to paste into Claude Code / Cursor

---

Build a multi-agent competitive intelligence system using LangGraph and the Anthropic Claude API. The system monitors a configurable list of AI companies and produces a structured weekly competitive briefing.

### Architecture

Three agents connected in a LangGraph state graph:

**Agent 1 — Researcher**
- Receives a list of company names as input
- For each company, uses the Tavily search tool to find news and developments from the last 7 days
- Searches for: product launches, funding announcements, partnerships, research publications, hiring signals, regulatory developments
- Returns structured data per company:
  ```
  {
    company: string,
    findings: [
      {
        headline: string,
        source_url: string,
        date: string,
        summary: string,
        category: "product_launch" | "funding" | "partnership" | "research" | "hiring" | "regulatory"
      }
    ]
  }
  ```
- If search returns no results or low-quality results for a finding, mark it with `confidence: "low"` and include a flag

**Agent 2 — Analyst**
- Receives all Researcher output across all companies
- For each finding, assigns strategic significance: `HIGH`, `MEDIUM`, or `LOW` with a one-sentence reasoning
- Drops any finding marked `confidence: "low"` by the Researcher — do not include unverified claims
- Identifies cross-company trends: look for themes appearing in 2 or more companies
- Produces a Watch List: 2-3 items that warrant close attention in the coming week
- Output is structured JSON ready for the Writer to consume

**Agent 3 — Writer**
- Receives Analyst output
- Writes a clean competitive briefing in this exact format:

```
WEEKLY COMPETITIVE INTELLIGENCE BRIEF
Week of {date}

EXECUTIVE SUMMARY
{3-4 sentences. What is the single most important thing that happened this week and why does it matter?}

COMPANY SNAPSHOTS
{For each company with findings:}
{Company Name}
• {headline} [{significance}]
• {headline} [{significance}]

CROSS-CUTTING TRENDS
→ {trend 1}
→ {trend 2}

WATCH LIST
! {item 1 — why it matters}
! {item 2 — why it matters}
```

- Only include HIGH and MEDIUM significance findings in the briefing
- If a company had no noteworthy developments, include one line: "{Company}: No significant developments this week."

---

### State schema

Define a LangGraph `TypedDict` state that carries:
- `companies`: list of company names to track
- `researcher_output`: raw findings per company
- `analyst_output`: scored and filtered findings with trends and watch list
- `final_briefing`: the finished markdown string
- `errors`: list of any failures (search failures, parsing errors) — surface these at the end, don't silently swallow them

---

### Graph structure

```
START → researcher_node → analyst_node → writer_node → END
```

Use `StateGraph` with a single linear flow for now. Each node is a function that takes state and returns updated state.

---

### Tools

Configure Tavily as a LangChain tool:
```python
from langchain_community.tools.tavily_search import TavilySearchResults
search_tool = TavilySearchResults(max_results=5)
```

Bind the search tool to the Researcher agent only. The Analyst and Writer use pure LLM calls with no tools.

---

### Model config

Use `claude-sonnet-4-20250514` for all three agents. Set temperature:
- Researcher: 0.1 (factual, consistent)
- Analyst: 0.2 (some judgment, mostly consistent)
- Writer: 0.4 (some variation in prose)

---

### Entry point

Create a `run_briefing(companies: list[str])` function that:
1. Initializes the graph state with the company list
2. Runs the full graph
3. Prints the final briefing to stdout
4. Saves it as `briefing_{YYYY-MM-DD}.md` in a `/output` folder
5. Prints any errors logged during the run

Default company list to use for testing:
```python
companies = ["OpenAI", "Anthropic", "Google DeepMind", "Mistral", "Cohere"]
```

---

### Error handling requirements

- If Tavily returns no results for a company, log the error to state and continue — don't crash
- If any agent's LLM call fails, retry once, then log and skip that step
- If the Analyst receives zero valid findings (all dropped due to low confidence), the Writer should produce a briefing that says so clearly rather than hallucinating content

---

### File structure to create

```
competitive_intel/
├── main.py              # entry point, run_briefing()
├── graph.py             # LangGraph state graph definition
├── agents/
│   ├── researcher.py    # Agent 1
│   ├── analyst.py       # Agent 2
│   └── writer.py        # Agent 3
├── tools/
│   └── search.py        # Tavily tool config
├── output/              # generated briefings saved here
└── requirements.txt
```

---

### Requirements.txt to generate

```
langgraph
langchain
langchain-anthropic
langchain-community
tavily-python
anthropic
python-dotenv
```

---

### Environment variables needed

```
ANTHROPIC_API_KEY=
TAVILY_API_KEY=
```

Load these from a `.env` file using `python-dotenv`.

---

## What to build first (sequence)

1. Get the Researcher working for a single company — confirm Tavily search returns real results
2. Confirm structured output parses cleanly into the state schema
3. Add the Analyst — test with the Researcher's output hardcoded so you can iterate fast
4. Add the Writer — test with hardcoded Analyst output
5. Wire all three into the LangGraph state graph
6. Run end to end with all 5 companies
7. Add the error handling layer last

---

## Stretch goals (once it's working)

- Add a 4th agent: a **Fact Checker** that takes any HIGH significance finding and does a second search to verify it before the Writer includes it — this is the governance/trustworthy AI angle
- Add Slack or email output so the briefing delivers itself
- Make the company list configurable via a simple CLI argument
- Add a "last week's briefing" input so the Analyst can flag what's changed vs. what's repeated

---

## Why this project matters for the Apple interview

This build demonstrates:
- Hands-on agentic AI architecture (LangGraph, multi-agent orchestration)
- Production thinking (error handling, confidence scoring, no hallucinated content)
- Governance instinct (low-confidence findings are dropped, not included)
- Direct alignment with the JD: "track the competitive landscape across GenAI products, agentic systems, orchestration frameworks"
- Vibe coding fluency — you built this yourself, not just described it
