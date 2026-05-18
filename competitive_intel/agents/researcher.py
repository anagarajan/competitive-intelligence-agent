import json
import re
from typing import Callable, Any

from langchain_core.tools.base import ToolException

from competitive_intel.config import Config
from competitive_intel.llm import get_llm
from competitive_intel.tools.search import get_search_tool


def _extract_results(raw: Any) -> list[dict]:
    """Normalize TavilySearch output to a flat list of result dicts.

    TavilySearch returns {query, results, images, ...} while the legacy
    TavilySearchResults returned a plain list. Both formats are handled here.
    """
    if isinstance(raw, dict):
        return raw.get("results", [])
    if isinstance(raw, list):
        return raw
    return []

_SYSTEM_PROMPT = """\
You are a competitive intelligence researcher specializing in the {industry} industry{niche_ctx}.

Given search results for a company, extract ONLY findings that are directly relevant to the \
{industry} industry{niche_ctx}. Ignore general company news (layoffs, unrelated lawsuits, \
product launches in other industries, etc.) unless they have a clear and specific impact on \
{industry}{niche_ctx}.

For each relevant finding return:
- headline: concise description of the development
- source_url: URL of the source (use the url field from search results)
- date: date of the development, or "recent" if unclear
- summary: 1-2 sentence summary explaining WHY this matters for {industry}{niche_ctx}
- category: one of product_launch, funding, partnership, research, hiring, regulatory
- confidence: "high" if well-sourced and specific, "low" if vague or unverified

Respond with ONLY valid JSON:
{{
  "company": "<official company name>",
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

If no findings are directly relevant to {industry}{niche_ctx}, return an empty findings array. \
Do not force-fit general news into the industry context.
"""

_CANONICALIZE_PROMPT = """\
Map each company name to its official full name. Return ONLY valid JSON: \
{"input_name": "Official Company Name"}.
If a name is already correct, keep it as-is. No extra text or code fences.\
"""


def _canonicalize_companies(llm, companies: list[str], industry: str) -> dict[str, str]:
    messages = [
        {"role": "system", "content": _CANONICALIZE_PROMPT},
        {"role": "user", "content": f"Industry: {industry}\nCompanies: {json.dumps(companies)}"},
    ]
    try:
        content = llm.invoke(messages).content
        content = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content.strip())
        mapping = json.loads(content)
        return {c: mapping.get(c, c) for c in companies}
    except Exception:
        return {c: c for c in companies}


def make_researcher_node(config: Config) -> Callable[[dict], dict]:
    llm = get_llm(config, temperature=0.1)
    search_tool = get_search_tool()
    niche_ctx = f", specifically {config.niche}" if config.niche else ""
    system_prompt = _SYSTEM_PROMPT.format(industry=config.industry, niche_ctx=niche_ctx)

    def researcher_node(state: dict) -> dict:
        companies: list[str] = state["companies"]
        errors: list[str] = list(state.get("errors", []))
        all_findings = []

        canonical = _canonicalize_companies(llm, companies, config.industry)

        for company in companies:
            canonical_name = canonical.get(company, company)
            niche_part = f" {config.niche}" if config.niche else ""

            queries = [
                f"{canonical_name} {config.industry}{niche_part} news developments 2026",
                f"{canonical_name} {config.industry}{niche_part} funding acquisition partnership product launch 2026",
            ]

            seen_urls: set[str] = set()
            aggregated_results: list = []
            search_failed = False

            for query in queries:
                try:
                    raw = search_tool.invoke(query)
                    for r in _extract_results(raw):
                        url = r.get("url", "")
                        if url not in seen_urls:
                            seen_urls.add(url)
                            aggregated_results.append(r)
                except ToolException:
                    pass  # no results for this query — try the next one
                except Exception as exc:
                    if not search_failed:
                        errors.append(f"Tavily search failed for '{company}': {exc}")
                        search_failed = True

            if search_failed and not aggregated_results:
                all_findings.append({"company": canonical_name, "findings": []})
                continue

            if not aggregated_results:
                all_findings.append({
                    "company": canonical_name,
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
                        f"Company: {canonical_name}\n"
                        f"Search results:\n{json.dumps(aggregated_results, indent=2)}"
                    ),
                },
            ]

            try:
                response = llm.invoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                content = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
                content = re.sub(r"\s*```$", "", content.strip())
                parsed = json.loads(content)
                all_findings.append(parsed)
            except json.JSONDecodeError as exc:
                errors.append(f"Failed to parse LLM response for '{company}': {exc}")
                all_findings.append({"company": canonical_name, "findings": []})
            except Exception as exc:
                errors.append(f"LLM call failed for '{company}': {exc}")
                all_findings.append({"company": canonical_name, "findings": []})

        return {**state, "researcher_output": all_findings, "errors": errors}

    return researcher_node
