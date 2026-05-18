import json
import pytest
from unittest.mock import MagicMock, patch
from competitive_intel.config import Config
from competitive_intel.agents.researcher import make_researcher_node


def _make_config(**kwargs):
    defaults = dict(industry="FinTech", companies=["Stripe"], provider="claude")
    return Config(**{**defaults, **kwargs})


def _mock_llm_response(findings: list, company="Stripe") -> MagicMock:
    response = MagicMock()
    response.content = json.dumps({"company": company, "findings": findings})
    return response


def _identity_canonical(llm, companies, industry):
    return {c: c for c in companies}


class TestResearcherNode:
    def _run_node(self, config, mock_llm, mock_search, state):
        """Build and run the node entirely within all patches."""
        with patch("competitive_intel.agents.researcher.get_llm", return_value=mock_llm), \
             patch("competitive_intel.agents.researcher.get_search_tool", return_value=mock_search), \
             patch("competitive_intel.agents.researcher._canonicalize_companies",
                   side_effect=_identity_canonical):
            node = make_researcher_node(config)
            return node(state)

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

        result = self._run_node(cfg, mock_llm, mock_search, {"companies": ["Stripe"], "errors": []})

        assert len(result["researcher_output"]) == 1
        assert result["researcher_output"][0]["company"] == "Stripe"
        assert len(result["researcher_output"][0]["findings"]) == 1
        assert result["errors"] == []

    def test_tavily_exception_on_all_searches_logs_error(self):
        cfg = _make_config(companies=["Stripe", "Plaid"])
        mock_llm = MagicMock()
        mock_search = MagicMock()
        # Both Stripe searches fail; both Plaid searches succeed
        mock_search.invoke.side_effect = [
            Exception("API timeout"),          # Stripe query 1
            Exception("API timeout"),          # Stripe query 2
            [{"url": "u", "content": "c"}],    # Plaid query 1
            [{"url": "u2", "content": "c2"}],  # Plaid query 2
        ]
        mock_llm.invoke.return_value = _mock_llm_response([], company="Plaid")

        result = self._run_node(cfg, mock_llm, mock_search, {"companies": ["Stripe", "Plaid"], "errors": []})

        assert len(result["errors"]) == 1
        assert "Stripe" in result["errors"][0]
        assert len(result["researcher_output"]) == 2

    def test_tavily_partial_failure_uses_remaining_results(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response([])
        mock_search = MagicMock()
        # First search fails but second succeeds — company still gets results
        mock_search.invoke.side_effect = [
            Exception("timeout"),
            [{"url": "u", "content": "c"}],
        ]

        result = self._run_node(cfg, mock_llm, mock_search, {"companies": ["Stripe"], "errors": []})

        assert len(result["errors"]) == 1
        assert mock_llm.invoke.called  # LLM still called with partial results

    def test_empty_search_results_marks_low_confidence(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_search = MagicMock()
        mock_search.invoke.return_value = []

        result = self._run_node(cfg, mock_llm, mock_search, {"companies": ["Stripe"], "errors": []})

        findings = result["researcher_output"][0]["findings"]
        assert any(f["confidence"] == "low" for f in findings)
        mock_llm.invoke.assert_not_called()

    def test_niche_included_in_query(self):
        cfg = _make_config(niche="payments infrastructure")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response([])
        mock_search = MagicMock()
        mock_search.invoke.return_value = [{"url": "u", "content": "c"}]

        self._run_node(cfg, mock_llm, mock_search, {"companies": ["Stripe"], "errors": []})

        for call in mock_search.invoke.call_args_list:
            assert "payments infrastructure" in call[0][0]

    def test_state_dict_returned_intact(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response([])
        mock_search = MagicMock()
        mock_search.invoke.return_value = [{"url": "u", "content": "c"}]

        result = self._run_node(
            cfg, mock_llm, mock_search,
            {"companies": ["Stripe"], "errors": [], "analyst_output": None, "final_briefing": ""},
        )

        assert "analyst_output" in result
        assert "final_briefing" in result

    def test_deduplicates_urls_across_searches(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response([])
        mock_search = MagicMock()
        same_result = [{"url": "http://same.com", "content": "c"}]
        mock_search.invoke.side_effect = [same_result, same_result]

        self._run_node(cfg, mock_llm, mock_search, {"companies": ["Stripe"], "errors": []})

        llm_call_content = mock_llm.invoke.call_args[0][0][1]["content"]
        results_in_prompt = json.loads(llm_call_content.split("Search results:\n")[1])
        assert len(results_in_prompt) == 1
