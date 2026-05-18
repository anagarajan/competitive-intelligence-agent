import pytest
from unittest.mock import MagicMock, patch
from competitive_intel.config import Config
from competitive_intel.graph import build_graph, IntelState


def _make_config(**kwargs):
    defaults = dict(industry="FinTech", companies=["Stripe"], provider="claude")
    return Config(**{**defaults, **kwargs})


def _mock_node(return_update: dict):
    """Return a node function that merges return_update into state."""
    def node(state):
        return {**state, **return_update}
    return node


class TestBuildGraph:
    def test_graph_compiles_without_error(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_search = MagicMock()
        with patch("competitive_intel.agents.researcher.get_llm", return_value=mock_llm), \
             patch("competitive_intel.agents.researcher.get_search_tool", return_value=mock_search), \
             patch("competitive_intel.agents.analyst.get_llm", return_value=mock_llm), \
             patch("competitive_intel.agents.writer.get_llm", return_value=mock_llm):
            graph = build_graph(cfg)
        assert graph is not None

    def test_full_graph_flows_through_all_nodes(self):
        cfg = _make_config()
        researcher_findings = [{"company": "Stripe", "findings": [
            {"headline": "h", "source_url": "u", "date": "d", "summary": "s",
             "category": "funding", "confidence": "high"}
        ]}]
        analyst_output = {
            "scored_findings": [{"company": "Stripe", "headline": "h", "significance": "HIGH", "reasoning": "r"}],
            "cross_company_trends": [], "watch_list": [], "zero_valid_findings": False,
        }
        mock_researcher_llm = MagicMock()
        import json
        mock_researcher_llm.invoke.return_value = MagicMock(
            content=json.dumps({"company": "Stripe", "findings": researcher_findings[0]["findings"]})
        )
        mock_analyst_llm = MagicMock()
        mock_analyst_llm.invoke.return_value = MagicMock(content=json.dumps(analyst_output))
        mock_writer_llm = MagicMock()
        mock_writer_llm.invoke.return_value = MagicMock(content="WEEKLY COMPETITIVE INTELLIGENCE BRIEF\n...")
        mock_search = MagicMock()
        mock_search.invoke.return_value = [{"url": "u", "content": "c"}]

        with patch("competitive_intel.agents.researcher.get_llm", return_value=mock_researcher_llm), \
             patch("competitive_intel.agents.researcher.get_search_tool", return_value=mock_search), \
             patch("competitive_intel.agents.analyst.get_llm", return_value=mock_analyst_llm), \
             patch("competitive_intel.agents.writer.get_llm", return_value=mock_writer_llm):
            graph = build_graph(cfg)
            result = graph.invoke({
                "companies": ["Stripe"],
                "researcher_output": [],
                "analyst_output": None,
                "final_briefing": "",
                "errors": [],
            })

        assert result["researcher_output"] != []
        assert result["analyst_output"] is not None
        assert "WEEKLY" in result["final_briefing"]

    def test_errors_from_researcher_preserved_in_final_state(self):
        cfg = _make_config()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"company": "Stripe", "findings": []}')
        mock_search = MagicMock()
        mock_search.invoke.side_effect = Exception("Tavily down")

        import json
        analyst_llm = MagicMock()
        analyst_llm.invoke.return_value = MagicMock(content=json.dumps({
            "scored_findings": [], "cross_company_trends": [], "watch_list": [], "zero_valid_findings": True
        }))
        writer_llm = MagicMock()
        writer_llm.invoke.return_value = MagicMock(content="Brief: no findings.")

        with patch("competitive_intel.agents.researcher.get_llm", return_value=mock_llm), \
             patch("competitive_intel.agents.researcher.get_search_tool", return_value=mock_search), \
             patch("competitive_intel.agents.analyst.get_llm", return_value=analyst_llm), \
             patch("competitive_intel.agents.writer.get_llm", return_value=writer_llm):
            graph = build_graph(cfg)
            result = graph.invoke({
                "companies": ["Stripe"],
                "researcher_output": [],
                "analyst_output": None,
                "final_briefing": "",
                "errors": [],
            })

        assert len(result["errors"]) >= 1
        assert result["final_briefing"] != ""
