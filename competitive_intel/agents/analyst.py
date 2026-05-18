import json
import re
from typing import Callable

from competitive_intel.config import Config
from competitive_intel.llm import get_llm

_SYSTEM_PROMPT = """\
You are a strategic intelligence analyst specializing in the {industry} industry{niche_ctx}.

You will receive research findings (already filtered for quality). Your tasks:
1. Score each finding's strategic significance: HIGH, MEDIUM, or LOW
2. Identify cross-company trends (themes appearing in 2+ companies)
3. Produce a Watch List of 2-3 items warranting close attention next week

Significance guidelines for {industry}:
- HIGH: major competitive moves, significant funding, game-changing partnerships or products
- MEDIUM: notable but incremental developments, strategic hires, new market entries
- LOW: routine announcements, minor updates

Respond with ONLY valid JSON:
{{
  "scored_findings": [
    {{
      "company": "...",
      "headline": "...",
      "significance": "HIGH",
      "reasoning": "One sentence explaining the score."
    }}
  ],
  "cross_company_trends": ["Trend description"],
  "watch_list": ["Item — why it matters"],
  "zero_valid_findings": false
}}
"""

_ZERO_FINDINGS_OUTPUT = {
    "scored_findings": [],
    "cross_company_trends": [],
    "watch_list": [],
    "zero_valid_findings": True,
}


def make_analyst_node(config: Config) -> Callable[[dict], dict]:
    llm = get_llm(config, temperature=0.2)
    niche_ctx = f", specifically {config.niche}" if config.niche else ""
    system_prompt = _SYSTEM_PROMPT.format(industry=config.industry, niche_ctx=niche_ctx)

    def analyst_node(state: dict) -> dict:
        errors: list[str] = list(state.get("errors", []))
        researcher_output: list = state.get("researcher_output", [])

        # Drop low-confidence findings before the LLM ever sees them
        valid_findings = [
            {
                "company": item["company"],
                "findings": [
                    f for f in item.get("findings", [])
                    if f.get("confidence") != "low"
                ],
            }
            for item in researcher_output
            if any(f.get("confidence") != "low" for f in item.get("findings", []))
        ]

        if not valid_findings:
            return {**state, "analyst_output": _ZERO_FINDINGS_OUTPUT, "errors": errors}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research findings:\n{json.dumps(valid_findings, indent=2)}"},
        ]

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = llm.invoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                content = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
                content = re.sub(r"\s*```$", "", content.strip())
                parsed = json.loads(content)
                return {**state, "analyst_output": parsed, "errors": errors}
            except Exception as exc:
                last_error = exc

        errors.append(f"Analyst LLM call failed after 2 attempts: {last_error}")
        return {**state, "analyst_output": _ZERO_FINDINGS_OUTPUT, "errors": errors}

    return analyst_node
