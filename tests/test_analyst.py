import json
import pytest
from unittest.mock import MagicMock, patch
from competitive_intel.config import Config
from competitive_intel.agents.analyst import make_analyst_node, _ZERO_FINDINGS_OUTPUT


def _make_config(**kwargs):
    defaults = dict(industry="FinTech", companies=["Stripe"], provider="claude")
    return Config(**{**defaults, **kwargs})


def _finding(company="Stripe", confidence="high", headline="Stripe raises $1B"):
    return {"company": company, "findings": [
        {"headline": headline, "source_url": "http://ex.com", "date": "2026-05-01",
         "summary": "Big round.", "category": "funding", "confidence": confidence}
    ]}


def _make_node(config, mock_llm):
    with patch("competitive_intel.agents.analyst.get_llm", return_value=mock_llm):
        return make_analyst_node(config)


class TestAnalystNode:
    def test_happy_path_returns_scored_output(self):
        cfg = _make_config()
        output = {
            "scored_findings": [{"company": "Stripe", "headline": "h", "significance": "HIGH", "reasoning": "r"}],
            "cross_company_trends": ["AI adoption"],
            "watch_list": ["Watch Stripe"],
            "zero_valid_findings": False,
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(output))

        node = _make_node(cfg, mock_llm)
        result = node({"researcher_output": [_finding()], "errors": []})

        assert result["analyst_output"]["scored_findings"][0]["significance"] == "HIGH"
        assert result["analyst_output"]["zero_valid_findings"] is False
        assert result["errors"] == []

    def test_low_confidence_findings_dropped(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "scored_findings": [], "cross_company_trends": [], "watch_list": [], "zero_valid_findings": True
        }))

        node = _make_node(cfg, mock_llm)
        # All findings are low confidence
        low_data = [_finding(confidence="low")]
        result = node({"researcher_output": low_data, "errors": []})

        assert result["analyst_output"]["zero_valid_findings"] is True
        mock_llm.invoke.assert_not_called()

    def test_all_low_confidence_returns_zero_findings(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        node = _make_node(cfg, mock_llm)
        result = node({"researcher_output": [_finding(confidence="low")], "errors": []})

        assert result["analyst_output"] == _ZERO_FINDINGS_OUTPUT
        mock_llm.invoke.assert_not_called()

    def test_empty_researcher_output_returns_zero_findings(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        node = _make_node(cfg, mock_llm)
        result = node({"researcher_output": [], "errors": []})

        assert result["analyst_output"]["zero_valid_findings"] is True
        mock_llm.invoke.assert_not_called()

    def test_single_company_cross_trends_can_be_empty(self):
        cfg = _make_config()
        output = {
            "scored_findings": [{"company": "Stripe", "headline": "h", "significance": "MEDIUM", "reasoning": "r"}],
            "cross_company_trends": [],
            "watch_list": ["Watch Stripe"],
            "zero_valid_findings": False,
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(output))
        node = _make_node(cfg, mock_llm)
        result = node({"researcher_output": [_finding()], "errors": []})

        assert result["analyst_output"]["cross_company_trends"] == []

    def test_llm_failure_retries_and_logs_error(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM timeout")
        node = _make_node(cfg, mock_llm)
        result = node({"researcher_output": [_finding()], "errors": []})

        assert mock_llm.invoke.call_count == 2
        assert result["analyst_output"]["zero_valid_findings"] is True
        assert len(result["errors"]) == 1
        assert "2 attempts" in result["errors"][0]

    def test_existing_errors_preserved(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "scored_findings": [], "cross_company_trends": [], "watch_list": [], "zero_valid_findings": True
        }))
        node = _make_node(cfg, mock_llm)
        result = node({"researcher_output": [], "errors": ["prior error"]})

        assert "prior error" in result["errors"]
