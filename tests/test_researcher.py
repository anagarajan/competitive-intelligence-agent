import json
import pytest
from unittest.mock import MagicMock, patch
from competitive_intel.config import Config
from competitive_intel.agents.researcher import make_researcher_node


def _make_config(**kwargs):
    defaults = dict(industry="FinTech", companies=["Stripe"], provider="claude")
    return Config(**{**defaults, **kwargs})


def _mock_llm_response(findings: list) -> MagicMock:
    response = MagicMock()
    response.content = json.dumps({"company": "Stripe", "findings": findings})
    return response


class TestResearcherNode:
    def _build_node(self, config, mock_llm, mock_search):
        with patch("competitive_intel.agents.researcher.get_llm", return_value=mock_llm), \
             patch("competitive_intel.agents.researcher.get_search_tool", return_value=mock_search):
            return make_researcher_node(config)

    def test_happy_path_returns_findings(self):
        cfg = _make_config()
        findings = [
            {"headline": "Stripe raises $1B", "source_url": "http://ex.com", "date": "2026-05-01",
             "summary": "Big round.", "category": "funding", "confidence": "high"}
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response(findings)
        mock_search = MagicMock()
        mock_search.invoke.return_value = [{"url": "http://ex.com", "content": "Stripe raises $1B"}]

        node = self._build_node(cfg, mock_llm, mock_search)
        result = node({"companies": ["Stripe"], "errors": []})

        assert len(result["researcher_output"]) == 1
        assert result["researcher_output"][0]["company"] == "Stripe"
        assert len(result["researcher_output"][0]["findings"]) == 1
        assert result["errors"] == []

    def test_tavily_exception_logs_error_continues(self):
        cfg = _make_config(companies=["Stripe", "Plaid"])
        mock_llm = MagicMock()
        mock_search = MagicMock()
        mock_search.invoke.side_effect = [Exception("API timeout"), [{"url": "u", "content": "c"}]]
        mock_llm.invoke.return_value = _mock_llm_response([])

        node = self._build_node(cfg, mock_llm, mock_search)
        result = node({"companies": ["Stripe", "Plaid"], "errors": []})

        assert len(result["errors"]) == 1
        assert "Stripe" in result["errors"][0]
        assert len(result["researcher_output"]) == 2

    def test_empty_search_results_marks_low_confidence(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_search = MagicMock()
        mock_search.invoke.return_value = []

        node = self._build_node(cfg, mock_llm, mock_search)
        result = node({"companies": ["Stripe"], "errors": []})

        findings = result["researcher_output"][0]["findings"]
        assert any(f["confidence"] == "low" for f in findings)
        mock_llm.invoke.assert_not_called()

    def test_niche_included_in_query(self):
        cfg = _make_config(niche="payments infrastructure")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response([])
        mock_search = MagicMock()
        mock_search.invoke.return_value = [{"url": "u", "content": "c"}]

        node = self._build_node(cfg, mock_llm, mock_search)
        node({"companies": ["Stripe"], "errors": []})

        call_args = mock_search.invoke.call_args[0][0]
        assert "payments infrastructure" in call_args

    def test_state_dict_returned_intact(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response([])
        mock_search = MagicMock()
        mock_search.invoke.return_value = [{"url": "u", "content": "c"}]

        node = self._build_node(cfg, mock_llm, mock_search)
        result = node({"companies": ["Stripe"], "errors": [], "analyst_output": None, "final_briefing": ""})

        assert "analyst_output" in result
        assert "final_briefing" in result
