from smolagents import DuckDuckGoSearchTool, tool, Tool
from loguru import logger
from geoagency.agents.retriever.retriever import RepoRetriever
from tomark import Tomark

class MetadataRetrieverTool(Tool):
    name = "retrieve_metadata"
    description = """
    This tool can be used to retrieve metadata from different repositories.
    It then shows the top 3 search results.
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
        
        for r in top_3_results:
            r.pop('meta')

        formatted_results = Tomark.table(top_3_results)
            
        return {
            "formatted_string": f"Here is a table including the top-3 search results:\n {formatted_results}",
            "structured_data": top_3_results
        }

@tool
def clarify_search_criteria(query: str) -> str:
    """
    Ask for clarification if the search criteria are not clear INSTEAD of conducting a search. 
    
    Consider spatial, thematic or temporal criteria depending on the context.
    
    Args:
        query: The original query string.
    """
    return "Please provide more information to clarify the search criteria."