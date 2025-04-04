from smolagents import tool, Tool
from loguru import logger
from geoagency.agents.retriever.retriever import RepoRetriever
# Import necessary for agent reference if needed, or handle state differently
# from __main__ import MetadataRetrieverAgent # Example if agent class is in the main script
from typing import Any, Dict, List, Optional, Tuple
import asyncio

# Keep the MetadataRetrieverTool largely the same as the previous CodeAgent revision
class MetadataRetrieverTool(Tool):
    name = "retrieve_metadata"
    description = """
    Retrieves dataset metadata from configured repositories based on a list of search queries.
    Use like: retrieve_metadata(queries=["search term 1", "query about topic X"])
    Returns a text summary of top results found for the agent to observe.
    Full structured results are stored internally for subsequent processing (like top-10 selection).
    """
    inputs = {
        "queries": {
            "type": "object",
            "description": "List with query strings to search for.",
        }
    }
    output_type = "string"

    def __init__(self, retriever: RepoRetriever):
        super().__init__()
        self.retriever = retriever
        self.last_full_results: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self.agent_instance = None

    def set_agent_instance(self, agent):
         self.agent_instance = agent

    def _format_summary(self, search_results: Dict[str, List[Dict[str, Any]]]) -> str:
        # (Implementation remains the same - unchanged)
        if not search_results:
            return "No results found for the given queries."
        summary_lines = []
        total_found = sum(len(v) for v in search_results.values())
        results_shown_count = 0
        max_summary_results = 5
        for query, results in search_results.items():
            if results:
                summary_lines.append(f"--- Top Results Summary for '{query}' ---")
                for result in results[:max_summary_results]:
                    if results_shown_count >= max_summary_results: break
                    title = result.get('text', 'No description')[:100]
                    source = result.get('source', '#')
                    result_id = result.get('id', 'N/A')
                    region = result.get('geometry_geojson', None)
                    region_info = f"Region: {region}" if region else "Region: Not specified"
                    summary_lines.append(f"- Title: {title}...\n  Source: {source}\n  ID: {result_id}\n  {region_info}")
                    results_shown_count += 1
            if results_shown_count >= max_summary_results: break
        if not summary_lines: return "No results found."
        summary_lines.append(f"\n(Summary includes top {results_shown_count} of {total_found} total results. Full results stored for potential top-10 selection.)")
        return "\n".join(summary_lines)


    async def _run_retrieval(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        # (Implementation remains the same - async retrieval - unchanged)
        try:
            response = await self.retriever.query_multiple_repos(queries=queries, limit=20)
            return response
        except Exception as e:
            logger.error(f"Error during async metadata retrieval call: {e}")
            return {}

    def forward(self, queries: List[str]) -> str:
        """
        Executes the metadata search synchronously, handling potential running event loops.
        Stores full results, and returns a summary string.
        """
        logger.info(f"Retrieving metadata for queries: {queries}")
        self.last_full_results = None # Reset before call
        response = {}

        import asyncio
        response = asyncio.run(self.retriever.query_multiple_repos(queries=queries, limit=5))

        # Store full results on the instance FOR THE CALLBACK TO PICK UP
        self.last_full_results = response
        count = sum(len(v) for v in response.values()) if response else 0
        logger.info(f"Stored {count} total results internally for callback.")

        # Format and return the summary string for the agent's observation
        formatted_summary = self._format_summary(response)
        logger.debug(f"Returning summary for agent observation: {formatted_summary}")
        return formatted_summary

# --- NEW TOOL ---
class SelectTop10ResultsTool(Tool):
    name = "select_top_10_results"
    description = """
    Selects exactly the top 10 most relevant search results from the full set previously retrieved.
    Requires the original user query to assess relevance.
    Use AFTER 'retrieve_metadata' has been successfully called.
    Example call: select_top_10_results(original_query="user's initial question")
    Returns a message indicating success. The structured top 10 list is stored internally.
    """
    inputs = {
        "original_query": {
            "type": "string",
            "description": "The original query submitted by the user.",
        }
        # Note: We are NOT taking full_results as input here, see forward() implementation
    }
    output_type = "string" # Output simple confirmation message

    def __init__(self):
        super().__init__()
        self.agent_instance = None # Will hold the agent instance
        self.last_top_10_results: Optional[List[Dict[str, Any]]] = None

    def set_agent_instance(self, agent):
         # Method for the agent to provide its instance reference during setup
         self.agent_instance = agent

    def _simple_relevance_logic(self, full_results: Dict[str, List[Dict[str, Any]]], query: str) -> List[Dict[str, Any]]:
        """
        Basic logic: Combine all results, maybe prioritize based on keywords, return top 10.
        (This can be replaced with more sophisticated LLM-based re-ranking if needed)
        """
        if not full_results:
            return []

        combined_results = []
        for q_results in full_results.values():
            combined_results.extend(q_results)

        # Simple approach: Assume retriever's order is decent, just take top 10 overall.
        # Add basic keyword check for demonstration (optional enhancement)
        query_keywords = set(query.lower().split())
        def score_result(result):
            text_score = 0
            text = result.get('text', '').lower()
            if text:
                text_score = sum(1 for keyword in query_keywords if keyword in text)
            # Could add more scoring based on title, etc.
            return text_score # Higher score is better

        # Sort primarily by score (desc), then rely on original order as tie-breaker (implicitly)
        # This is still very basic relevance.
        try:
            # Add index to preserve original relative order for stability if scores are equal
            indexed_results = list(enumerate(combined_results))
            sorted_results_with_indices = sorted(indexed_results, key=lambda item: score_result(item[1]), reverse=True)
            # Extract original results in the new order
            sorted_results = [item[1] for item in sorted_results_with_indices]
        except Exception as e:
            logger.warning(f"Could not apply keyword scoring, falling back to original order: {e}")
            sorted_results = combined_results # Fallback

        return sorted_results[:10] # Return exactly top 10 (or fewer if less available)

    def forward(self, original_query: str) -> str:
        """
        Selects top 10 results based on relevance to the original query.
        Relies on accessing the agent's state for full results.
        """
        logger.info(f"Selecting top 10 results relevant to query: '{original_query}'")
        if not self.agent_instance:
             logger.error("Agent instance reference not set in SelectTop10ResultsTool.")
             return "Error: Tool configuration issue (agent instance missing)."
        if not hasattr(self.agent_instance, 'full_search_results') or not self.agent_instance.full_search_results:
             logger.warning("No full search results found on the agent instance to select from.")
             return "Warning: No search results available to select from. Did 'retrieve_metadata' run successfully?"

        # Access full results stored on the agent instance
        full_results = self.agent_instance.full_search_results

        # Apply relevance logic
        top_10 = self._simple_relevance_logic(full_results, original_query)

        # Store for the callback
        self.last_top_10_results = top_10
        logger.info(f"Stored {len(top_10)} results in last_top_10_results.")

        return f"Successfully selected and stored the top {len(top_10)} most relevant results."

# check_search_criteria_completeness tool remains the same
@tool
def check_search_criteria_completeness(search_criteria_spatial: str = None,
                                       search_criteria_temporal: str = None,
                                       search_criteria_thematic: str = None,
                                       spatial_criteria_necessary: bool = None,
                                       temporal_criteria_necessary: bool = None) -> str:
    """
    Validates if the search criteria provided or inferred from the query are sufficient.
    The agent should infer the necessary components based on the user's request.

    Args:
        search_criteria_spatial: The spatial component identified (e.g., "Germany", "bbox [...]").
        search_criteria_temporal: The temporal component identified (e.g., "from 2020 onwards", "specific date").
        search_criteria_thematic: The core topic identified (e.g., "precipitation", "soil moisture").
        spatial_criteria_necessary: Agent's assessment if spatial info is crucial for this query (True/False).
        temporal_criteria_necessary: Agent's assessment if temporal info is crucial for this query (True/False).

    Returns:
        A message indicating whether the search criteria are complete or suggesting what is missing or unclear.
    """
    missing_criteria = []
    if not search_criteria_thematic:
        missing_criteria.append("thematic information (what kind of data?)")

    if spatial_criteria_necessary and not search_criteria_spatial:
        missing_criteria.append("spatial information (where?)")

    if temporal_criteria_necessary and not search_criteria_temporal:
        missing_criteria.append("temporal information (when?)")

    if not missing_criteria:
        return "Search criteria seem complete. Proceed with retrieval."
    else:
        return f"Search criteria incomplete or unclear. Please clarify or provide the following: {', '.join(missing_criteria)}."