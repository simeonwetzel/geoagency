from smolagents import DuckDuckGoSearchTool, tool
from loguru import logger
from ..retriever.retriever import RepoRetriever

@tool
def retrieve_metadata(query: str, retriever: RepoRetriever) -> dict:
    """
    This tool can be used to retrieve metadata from different repositories.

    The output can be used to briefly summarize the top 5 results from the query.

    Args:
        query: The query string to search for.
        retriever: An instance of the RepoRetriever class.
    """
    import asyncio
    response = asyncio.run(retriever.query_all_repos(query))

    top_5_results = response.get("results", [])[:5]
    return top_5_results

@tool
def format_metadata_results(results: dict) -> str:
    """
    This tool can be used to format the metadata results.

    Also describe the metadata results.

    Args:
        results: The metadata results to format.
    """
    formatted_results = ""
    for idx, result in enumerate(results):
        formatted_results += f"Result {idx + 1}:\n"
        formatted_results += f"Text: {result.get('text', 'N/A')}\n"
        formatted_results += f"Source Repository: {result.get('source', 'N/A')}\n"

    return formatted_results