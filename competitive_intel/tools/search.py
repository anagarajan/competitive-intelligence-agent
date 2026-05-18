from langchain_tavily import TavilySearch


def get_search_tool() -> TavilySearch:
    return TavilySearch(max_results=5, topic="news")
