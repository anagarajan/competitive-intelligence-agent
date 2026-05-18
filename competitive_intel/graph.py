from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from competitive_intel.agents.analyst import make_analyst_node
from competitive_intel.agents.researcher import make_researcher_node
from competitive_intel.agents.writer import make_writer_node
from competitive_intel.config import Config


class IntelState(TypedDict):
    companies: list[str]
    researcher_output: list
    analyst_output: Optional[dict]
    final_briefing: str
    errors: list[str]


def build_graph(config: Config):
    graph = StateGraph(IntelState)

    graph.add_node("researcher", make_researcher_node(config))
    graph.add_node("analyst", make_analyst_node(config))
    graph.add_node("writer", make_writer_node(config))

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)

    return graph.compile()
