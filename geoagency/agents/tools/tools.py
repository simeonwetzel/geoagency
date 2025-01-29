from smolagents import DuckDuckGoSearchTool, tool, Tool
from loguru import logger
from ..retriever.retriever import RepoRetriever

class MetadataRetrieverTool(Tool):
    name = "retrieve_metadata"
    description = """
    This tool can be used to retrieve metadata from different repositories.
    The output can be used to briefly summarize the top 3 results from the query.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The query string to search for.",
        }
    }
    output_type = "object"
    def __init__(self, retriever: RepoRetriever):
        super().__init__()
        self.retriever = retriever

    def forward(self, query: str) -> dict:
        import asyncio
        response = asyncio.run(self.retriever.query_all_repos(query))

        top_3_results = response.get("results", [])[:3]
        
        formatted_results = ""
        for idx, result in enumerate(top_3_results):
            formatted_results += f"Result {idx + 1}:\n"
            formatted_results += f"Text: {result.get('text', 'N/A')}\n"
            formatted_results += f"Source Repository: {result.get('source', 'N/A')}\n"
            
        return formatted_results


@tool
def clarify_search_criteria(query: str) -> str:
    """
    Ask for clarification if the search criteria are not clear instead of conducting a search. 
    
    Consider spatial, thematic or temporal criteria depending on the context.
    
    Args:
        query: The original query string.
    """
    return "Please provide more information to clarify the search criteria."