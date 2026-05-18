import json
from typing import Callable

from competitive_intel.config import Config
from competitive_intel.llm import get_llm
from competitive_intel.tools.search import get_search_tool

_SYSTEM_PROMPT = """\
You are a competitive intelligence researcher specializing in the {industry} industry{niche_ctx}.

Given search results for a company, extract structured competitive intelligence findings.

For each finding return:
- headline: concise description of the development
- source_url: URL of the source (use the url field from search results)
- date: date of the development, or "recent" if unclear
- summary: 1-2 sentence summary
- category: one of product_launch, funding, partnership, research, hiring, regulatory
- confidence: "high" if well-sourced and specific, "low" if vague or unverified

Respond with ONLY valid JSON:
{{
  "company": "<company name>",
  "findings": [
    {{
      "headline": "...",
      "source_url": "...",
      "date": "...",
      "summary": "...",
      "category": "...",
      "confidence": "high"
    }}
  ]
}}

If no relevant findings exist, return an empty findings array.
"""


def make_researcher_node(config: Config) -> Callable[[dict], dict]:
    llm = get_llm(config, temperature=0.1)
    search_tool = get_search_tool()
    niche_ctx = f", specifically {config.niche}" if config.niche else ""
    system_prompt = _SYSTEM_PROMPT.format(industry=config.industry, niche_ctx=niche_ctx)

    def researcher_node(state: dict) -> dict:
        companies: list[str] = state["companies"]
        errors: list[str] = list(state.get("errors", []))
        all_findings = []

        for company in companies:
            niche_part = f" {config.niche}" if config.niche else ""
            query = f"{company} {config.industry}{niche_part} news developments last 7 days"

            try:
                results = search_tool.invoke(query)
            except Exception as exc:
                errors.append(f"Tavily search failed for '{company}': {exc}")
                all_findings.append({"company": company, "findings": []})
                continue

            if not results:
                all_findings.append({
                    "company": company,
                    "findings": [{"confidence": "low", "headline": "No results found",
                                  "source_url": "", "date": "recent", "summary": "",
                                  "category": "product_launch"}],
                })
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Company: {company}\n"
                        f"Search results:\n{json.dumps(results, indent=2)}"
                    ),
                },
            ]

            try:
                response = llm.invoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                parsed = json.loads(content)
                all_findings.append(parsed)
            except json.JSONDecodeError as exc:
                errors.append(f"Failed to parse LLM response for '{company}': {exc}")
                all_findings.append({"company": company, "findings": []})
            except Exception as exc:
                errors.append(f"LLM call failed for '{company}': {exc}")
                all_findings.append({"company": company, "findings": []})

        return {**state, "researcher_output": all_findings, "errors": errors}

    return researcher_node
