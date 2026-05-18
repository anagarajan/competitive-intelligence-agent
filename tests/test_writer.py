import pytest
from unittest.mock import MagicMock, patch
from competitive_intel.config import Config
from competitive_intel.agents.writer import make_writer_node

_ANALYST_OUTPUT = {
    "scored_findings": [
        {"company": "Stripe", "headline": "Stripe raises $1B", "significance": "HIGH", "reasoning": "Large round."},
        {"company": "Stripe", "headline": "Minor UI update", "significance": "LOW", "reasoning": "Small."},
        {"company": "Plaid", "headline": "Plaid partners with Chase", "significance": "MEDIUM", "reasoning": "Noteworthy."},
    ],
    "cross_company_trends": ["Embedded finance growth"],
    "watch_list": ["Watch Stripe — may signal expansion"],
    "zero_valid_findings": False,
}


def _make_config(**kwargs):
    defaults = dict(industry="FinTech", companies=["Stripe", "Plaid"], provider="claude")
    return Config(**{**defaults, **kwargs})


def _make_node(config, briefing_text):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=briefing_text)
    with patch("competitive_intel.agents.writer.get_llm", return_value=mock_llm):
        return make_writer_node(config), mock_llm


class TestWriterNode:
    def test_happy_path_returns_briefing(self):
        cfg = _make_config()
        expected = "WEEKLY COMPETITIVE INTELLIGENCE BRIEF\nWeek of May 17, 2026\n\nEXECUTIVE SUMMARY\nStripe had a big week.\n"
        node, _ = _make_node(cfg, expected)
        result = node({
            "companies": ["Stripe", "Plaid"],
            "analyst_output": _ANALYST_OUTPUT,
            "errors": [],
        })
        assert result["final_briefing"] == expected
        assert result["errors"] == []

    def test_briefing_includes_required_sections(self):
        cfg = _make_config()
        briefing = (
            "WEEKLY COMPETITIVE INTELLIGENCE BRIEF\nWeek of May 17, 2026\n\n"
            "EXECUTIVE SUMMARY\nSummary here.\n\n"
            "COMPANY SNAPSHOTS\nStripe\n• Stripe raises $1B [HIGH]\n\n"
            "CROSS-CUTTING TRENDS\n→ Embedded finance growth\n\n"
            "WATCH LIST\n! Watch Stripe — may signal expansion\n"
        )
        node, _ = _make_node(cfg, briefing)
        result = node({"companies": ["Stripe", "Plaid"], "analyst_output": _ANALYST_OUTPUT, "errors": []})

        for section in ["EXECUTIVE SUMMARY", "COMPANY SNAPSHOTS", "CROSS-CUTTING TRENDS", "WATCH LIST"]:
            assert section in result["final_briefing"]

    def test_zero_valid_findings_produces_briefing(self):
        cfg = _make_config()
        zero_output = {"scored_findings": [], "cross_company_trends": [], "watch_list": [], "zero_valid_findings": True}
        briefing = "WEEKLY COMPETITIVE INTELLIGENCE BRIEF\n...\nNo verifiable developments found this week.\n"
        node, _ = _make_node(cfg, briefing)
        result = node({"companies": ["Stripe"], "analyst_output": zero_output, "errors": []})

        assert result["final_briefing"] == briefing
        assert result["errors"] == []

    def test_llm_failure_returns_fallback_briefing(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        with patch("competitive_intel.agents.writer.get_llm", return_value=mock_llm):
            node = make_writer_node(cfg)
        result = node({"companies": ["Stripe"], "analyst_output": _ANALYST_OUTPUT, "errors": []})

        assert "ERROR" in result["final_briefing"]
        assert len(result["errors"]) == 1
        assert "Writer LLM" in result["errors"][0]

    def test_existing_errors_preserved(self):
        cfg = _make_config()
        node, _ = _make_node(cfg, "brief")
        result = node({"companies": ["Stripe"], "analyst_output": _ANALYST_OUTPUT, "errors": ["prior"]})
        assert "prior" in result["errors"]

    def test_none_analyst_output_handled(self):
        cfg = _make_config()
        node, _ = _make_node(cfg, "brief")
        # Should not raise even if analyst_output is None
        result = node({"companies": ["Stripe"], "analyst_output": None, "errors": []})
        assert "final_briefing" in result
