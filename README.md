# Competitive Intelligence Agent

A multi-agent system that searches the web for recent developments about companies in any industry and produces a structured weekly briefing. Built with [LangGraph](https://github.com/langchain-ai/langgraph), Claude or OpenAI, and Tavily Search.

## How it works

```
Researcher → Analyst → Writer
```

- **Researcher** — runs two targeted Tavily searches per company, canonicalizes company names, and extracts findings relevant to the specified industry/niche.
- **Analyst** — scores each finding (HIGH / MEDIUM / LOW), drops low-confidence results, and identifies cross-company trends.
- **Writer** — produces a formatted weekly brief with only HIGH and MEDIUM findings.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your API keys
```

**Required API keys** (in `.env`):

| Key | Required |
|-----|----------|
| `TAVILY_API_KEY` | Always |
| `ANTHROPIC_API_KEY` | When using `--provider claude` (default) |
| `OPENAI_API_KEY` | When using `--provider openai` |

Get keys at [tavily.com](https://tavily.com), [console.anthropic.com](https://console.anthropic.com), or [platform.openai.com](https://platform.openai.com).

## Usage

```bash
python -m competitive_intel.main \
  --industry "Threat detection and response" \
  --companies CrowdStrike SentinelOne "Microsoft Defender" \
  --niche "endpoint security" \
  --provider claude
```

| Flag | Description |
|------|-------------|
| `--industry` | Industry to monitor (**required**) |
| `--companies` | One or more companies to track (**required**) |
| `--niche` | Narrows focus within the industry (optional) |
| `--provider` | `claude` (default) or `openai` |
| `--model` | Override the default model for the chosen provider |

The briefing is printed to stdout and saved to `output/briefing_YYYY-MM-DD.md`.

## Output format

```
WEEKLY COMPETITIVE INTELLIGENCE BRIEF
Week of May 18, 2026

EXECUTIVE SUMMARY
...

COMPANY SNAPSHOTS
CrowdStrike
• CrowdStrike acquires SGNL to strengthen identity security [HIGH]

CROSS-CUTTING TRENDS
→ AI integration across endpoint security platforms

WATCH LIST
! CrowdStrike/SGNL integration — watch for product rollout timeline
```

## Running tests

```bash
pytest tests/
```

## Project structure

```
competitive_intel/
├── main.py          # CLI entry point
├── config.py        # Config dataclass
├── graph.py         # LangGraph pipeline definition
├── llm.py           # LLM factory (Claude / OpenAI)
├── agents/
│   ├── researcher.py
│   ├── analyst.py
│   └── writer.py
└── tools/
    └── search.py    # Tavily search tool
tests/               # pytest test suite (44 tests)
output/              # generated briefings (gitignored)
```
