import json
from datetime import date
from typing import Callable

from competitive_intel.config import Config
from competitive_intel.llm import get_llm

_SYSTEM_PROMPT = """\
You are a competitive intelligence writer producing weekly briefings for leaders in the {industry} industry.

Write in this EXACT format — do not deviate from the structure, symbols, or section names:

WEEKLY COMPETITIVE INTELLIGENCE BRIEF
Week of {{week_date}}

EXECUTIVE SUMMARY
{{3-4 sentences. What is the single most important development this week and why does it matter?}}

COMPANY SNAPSHOTS
{{Company Name}}
• {{headline}} [{{significance}}]

CROSS-CUTTING TRENDS
→ {{trend}}

WATCH LIST
! {{item — why it matters}}

Rules:
- Include only HIGH and MEDIUM significance findings
- If a company has no HIGH or MEDIUM findings, write exactly one line: "{{Company}}: No significant developments this week."
- Use • for findings, → for trends, ! for watch list items
- If zero_valid_findings is true, state that no verifiable developments were found this week — do not invent findings
- Return only the briefing text, no commentary or markdown fences
"""


def make_writer_node(config: Config) -> Callable[[dict], dict]:
    llm = get_llm(config, temperature=0.4)
    system_prompt = _SYSTEM_PROMPT.format(industry=config.industry)

    def writer_node(state: dict) -> dict:
        errors: list[str] = list(state.get("errors", []))
        analyst_output = state.get("analyst_output") or {}
        companies = state.get("companies", [])
        week_date = date.today().strftime("%B %d, %Y")

        payload = {
            "week_date": week_date,
            "companies": companies,
            "analyst_output": analyst_output,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Write the briefing for this week.\n\nData:\n{json.dumps(payload, indent=2)}",
            },
        ]

        try:
            response = llm.invoke(messages)
            briefing = response.content if hasattr(response, "content") else str(response)
            return {**state, "final_briefing": briefing, "errors": errors}
        except Exception as exc:
            errors.append(f"Writer LLM call failed: {exc}")
            fallback = (
                f"WEEKLY COMPETITIVE INTELLIGENCE BRIEF\n"
                f"Week of {week_date}\n\n"
                f"ERROR: Unable to generate briefing due to a technical failure. Please retry.\n"
            )
            return {**state, "final_briefing": fallback, "errors": errors}

    return writer_node
