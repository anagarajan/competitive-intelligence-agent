from langchain_community.tools.tavily_search import TavilySearchResults


def get_search_tool() -> TavilySearchResults:
    return TavilySearchResults(max_results=5)
