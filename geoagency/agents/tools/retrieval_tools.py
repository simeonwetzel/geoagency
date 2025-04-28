import logging
from smolagents import tool, Tool
from loguru import logger
from geoagency.agents.retriever.retriever import RepoRetriever
from typing import Any, Dict, List, Optional
import json


class MetadataRetrieverTool(Tool):
    name = "retrieve_metadata"
    description = """
    Retrieve metadata for datasets relevant to a user query. 
    This tool collects metadata from multiple repositories using RepoRetriever. 
    It is the first step in the search process and should be called before selecting top results.
    
    Instructions:
    - Create a list of at least 5 queries to search for datasets.    
    - Never use operators like 'AND', 'OR', 'NOT' in the queries, as they are not supported.
    
    Example: retrieve_metadata(queries=["climate data", "temperature records", "sea level measurements"])
    
    The output is a result dict with keys for all queries. The values for each of these keys are a list of dicts including the results for the certain query. 
    """

    inputs = {
        "queries": {
            "type": "object",
            "description": "List of query strings to search for datasets."
        }
    }

    output_type = "object"

    def __init__(self, retriever: RepoRetriever):
        super().__init__()
        self.retriever = retriever
        self.last_full_results: Optional[List[Dict[str, Any]]] = None

    def forward(self, queries: List[str]) -> str:
        """
        Calls the retriever to collect metadata for the query.
        Stores the results and returns a short summary.
        """
        logger.info(f"Retrieving metadata for queries: {queries}")

        try:
            # Call the retriever (assume it returns a list of dicts)
            import asyncio
            results = asyncio.run(
                self.retriever.query_multiple_repos(queries=queries, limit=4))

            if not results:
                self.last_full_results = []
                logger.warning("No results found for the query.")
                return "No datasets found matching the query."

            self.last_full_results = results

            count = sum(len(v) for v in results.values()) if results else 0
            logger.info(f"Retrieved {count} metadata records.")
            return results

        except Exception as e:
            logger.exception("Failed to retrieve metadata.")
            return f"Error during metadata retrieval: {str(e)}"


logger = logging.getLogger(__name__)


class SelectTop10ResultsTool(Tool):
    name = "select_top_10_results"
    description = """
    Selects and formats the top 10 most relevant datasets based on metadata search results collected previously.
    It uses an LLM to rank and select from the full results based on the original user query.

    Always call `retrieve_metadata` first to populate the necessary results.
    This tool handles both selection and formatting.

    The output is a dictionary containing:
    - 'message': A status message.
    - 'results': A list of the top 10 selected result dictionaries (structured data).
    - 'formatted_results': A string containing the formatted top 10 results, ready for display to the user.

    Example usage (Agent code generation):
    select_top_10_results(original_query="climate data germany")
    """
    inputs = {
        "original_query": {
            "type": "string",
            "description": "The original query provided by the user, used for relevance ranking.",
        },
        # Removed selected_results as input - relying on LLM selection based on full results
        # If manual selection is needed, it would require a different logic flow or tool.
    }
    output_type = "object"  # Returning an object with message, results list, and formatted string

    def __init__(self, llm: Any): # LLM is required for this tool's primary function now
        """
        Initializes the tool with an LLM for automatic selection and ranking.
        Requires access to the full results gathered by MetadataRetrieverTool via agent state.
        """
        super().__init__()
        if llm is None:
             # Making LLM mandatory for this tool's logic
             raise ValueError("SelectTop10ResultsTool requires an LLM instance.")
        self.llm = llm
        # These attributes are primarily for the callback mechanism in the Agent class
        self.last_top_10_results: Optional[List[Dict[str, Any]]] = None
        # This tool now relies on the agent providing the full results implicitly via state
        # or potentially passed explicitly if agent design changes. Let's assume agent passes it.
        # For smolagents, the agent usually passes relevant context or the tool fetches from agent state if needed.
        # We'll modify the agent's `run` or prompt to ensure results are available.
        # For simplicity here, let's assume the agent's full_search_results are accessible
        # This requires the agent instance or the results to be passed.
        # Let's refine this: the tool *shouldn't* rely on implicit agent state.
        # The agent should pass the necessary data.

    # Revised forward method signature if agent passes data:
    # def forward(self, original_query: str, full_results_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:

    # Assuming the agent framework makes the full_results_dict available somehow (e.g. via a context arg or agent reference):
    # Let's stick to the original idea that the agent runs retrieve_metadata first, and *then* runs this tool.
    # This tool *could* access the agent's state if set up (using set_agent_instance pattern),
    # or the agent could pass the flattened list in the call.
    # Let's assume the agent needs to pass the flattened list derived from its self.full_search_results.
    # MODIFYING TOOL SIGNATURE AND AGENT PROMPT ACCORDINGLY

    # --> Change Tool Description/Inputs
    description = """
    Selects and formats the top 10 most relevant datasets from a provided list of results.
    It uses an LLM to rank and select based on the original user query.

    **Requires a flattened list of all search results obtained from `retrieve_metadata`.**

    The output is a dictionary containing:
    - 'message': A status message.
    - 'results': A list of the top 10 selected result dictionaries (structured data).
    - 'formatted_results': A string containing the formatted top 10 results, ready for display to the user.

    Example usage (Agent code generation):
    # Assumes 'all_results_list' was previously created by flattening output from 'retrieve_metadata'
    select_top_10_results(original_query="climate data germany", all_results=all_results_list)
    """
    inputs = {
        "original_query": {
            "type": "string",
            "description": "The original query provided by the user, used for relevance ranking.",
        },
        "all_results": {
            "type": "object", # Should be array of objects
            "description": "A flattened list containing ALL metadata result dictionaries gathered previously.",
        }
    }

    def forward(self, original_query: str, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Uses an LLM to select the top 10 most relevant results from the provided list
        based on the original query, and formats them.
        """
        logger.info(f"Selecting top 10 results for query: '{original_query}' from {len(all_results)} candidates.")

        if not all_results:
            error_msg = "Error: No search results provided to select from. Ensure `retrieve_metadata` ran successfully and results were passed."
            logger.error(error_msg)
            self.last_top_10_results = []
            return {"message": error_msg, "results": [], "formatted_results": "No results to display."}

        # Prepare data for LLM - maybe select key fields to reduce prompt size
        # For now, send essential parts of the first N results (e.g., 50) to avoid overly large prompts
        max_results_for_llm = 50
        results_subset = all_results[:max_results_for_llm]

        # Extract relevant text for ranking (handle potential missing keys)
        def extract_text(res):
            title = res.get('meta', {}).get('title', [''])[0] if isinstance(res.get('meta', {}).get('title'), list) else res.get('meta', {}).get('title', '')
            desc = res.get('meta', {}).get('description', [''])[0] if isinstance(res.get('meta', {}).get('description'), list) else res.get('meta', {}).get('description', '')
            text_field = res.get('text', '') # The combined field from retriever
            # Combine relevant fields for the LLM to consider
            combined = f"ID: {res.get('id', 'N/A')}\nTitle: {title}\nDescription: {desc}\nDetails: {text_field}"
            return combined

        results_for_prompt = [extract_text(res) for res in results_subset]
        # Include original index to map back LLM selection to full result dict
        indexed_results_for_prompt = [{"original_index": i, "content": text} for i, text in enumerate(results_for_prompt)]

        try:
            prompt = f"""
You are an expert relevance ranker for scientific datasets.
Based on the user query: '{original_query}'

Review the following dataset summaries (limited to the first {len(indexed_results_for_prompt)}):
{json.dumps(indexed_results_for_prompt, indent=2)}

Select the top 10 most relevant datasets. Return ONLY a JSON list containing the 'original_index' of the top 10 selected items, ordered from most to least relevant.
Example Response Format: [5, 23, 1, 15, 49, 0, 8, 12, 33, 4]
"""
            # Ensure your LLMManager call syntax is correct
            llm_response_raw = self.llm.chat(prompt=prompt) # Adjust based on your LLMManager method
            llm_response_content = llm_response_raw # Assuming direct content string, adjust if nested

            logger.debug(f"LLM response for ranking: {llm_response_content}")

            # Parse the LLM response (expecting a list of indices)
            try:
                selected_indices = json.loads(llm_response_content)
                if not isinstance(selected_indices, list) or not all(isinstance(i, int) for i in selected_indices):
                     raise ValueError("LLM response is not a valid list of integers.")
            except (json.JSONDecodeError, ValueError) as parse_error:
                logger.error(f"Failed to parse LLM ranking response: {parse_error}")
                logger.error(f"LLM Raw Response was: {llm_response_content}")
                # Fallback or error handling: maybe take the first 10 raw results?
                selected_indices = list(range(min(10, len(all_results)))) # Fallback to first 10
                error_msg = f"Error parsing LLM ranking, using first {len(selected_indices)} results as fallback."
                logger.warning(error_msg)


            # Map indices back to the original full results dictionaries
            # Ensure indices are within the bounds of the subset used
            valid_indices = [idx for idx in selected_indices if 0 <= idx < len(results_subset)]
            top_10_selection = [results_subset[idx] for idx in valid_indices][:10] # Get top 10 valid


            self.last_top_10_results = top_10_selection # Store the structured results
            formatted_string = self._format_results(top_10_selection) # Format the selected results

            return {
                "message": f"Successfully selected and formatted top {len(top_10_selection)} results using LLM.",
                "results": self.last_top_10_results,
                "formatted_results": formatted_string
            }

        except Exception as e:
            logger.exception("LLM automatic selection failed.")
            # Fallback: return first 10 raw results if available
            fallback_results = all_results[:10]
            self.last_top_10_results = fallback_results
            formatted_fallback = self._format_results(fallback_results)
            return {
                "message": f"Error during automatic selection: {str(e)}. Returning first {len(fallback_results)} raw results as fallback.",
                "results": fallback_results,
                "formatted_results": formatted_fallback
            }

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Formats the selected top results into a user-friendly string.
        Attempts to extract common metadata fields, handling potential nesting and missing keys.
        """
        if not results:
            return "No results selected or available to format."

        formatted_string = "Here are the top relevant datasets found:\n\n"
        for i, result in enumerate(results):
            # Use .get() with defaults and handle potential list wrapping
            meta = result.get('meta', {}) if isinstance(result.get('meta'), dict) else {}

            # Safely extract title
            title_raw = meta.get('title', ['N/A'])
            title = title_raw[0] if isinstance(title_raw, list) and title_raw else str(title_raw)

            # Safely extract source (using homepage as link/source)
            source_raw = meta.get('homepage', ['N/A'])
            source = source_raw[0] if isinstance(source_raw, list) and source_raw else str(source_raw)
            # Fallback to 'source' field if 'homepage' is missing
            if source == 'N/A':
                 source_raw = result.get('source', ['N/A']) # Check top level
                 source = source_raw[0] if isinstance(source_raw, list) and source_raw else str(source_raw)

            # Safely extract description
            desc_raw = meta.get('description', ['No description available.'])
            description = desc_raw[0] if isinstance(desc_raw, list) and desc_raw else str(desc_raw)

            # Safely extract ID
            result_id = result.get('id', 'N/A')

            formatted_string += f"--- Result {i + 1} ---\n"
            formatted_string += f"  ID: {result_id}\n"
            formatted_string += f"  Title: {title}\n"
            formatted_string += f"  Link/Source: {source}\n"
            formatted_string += f"  Description: {description}\n\n"

        return formatted_string.strip()



@tool
def check_search_criteria_completeness(search_criteria_spatial: str = None,
                                       search_criteria_temporal: str = None,
                                       search_criteria_thematic: str = None,
                                       spatial_criteria_helpful: bool = None,
                                       temporal_criteria_helpful: bool = None) -> str:
    """
    Validates if the search criteria provided or inferred from the query are sufficient.
    The agent should infer the necessary components based on the user's request.

    Args:
        search_criteria_spatial: The spatial component identified (e.g., "Germany", "bbox [...]").
        search_criteria_temporal: The temporal component identified (e.g., "from 2020 onwards", "specific date").
        search_criteria_thematic: The core topic identified (e.g., "precipitation", "soil moisture").
        spatial_criteria_helpful: Agent's assessment if further spatial context could improve the search for this query (True/False).
        temporal_criteria_helpful: Agent's assessment if further temporal context could improve the search for this query (True/False).

    Returns:
        A message indicating whether the search criteria are complete or suggesting what is missing or unclear.
    """
    missing_criteria = []
    if not search_criteria_thematic:
        missing_criteria.append("thematic information (what kind of data?)")

    if spatial_criteria_helpful and not search_criteria_spatial:
        missing_criteria.append("spatial information (where?)")

    if temporal_criteria_helpful and not search_criteria_temporal:
        missing_criteria.append("temporal information (when?)")

    if not missing_criteria:
        return "Search criteria seem complete. Proceed with retrieval."
    else:
        return f"Search criteria incomplete or unclear. Please clarify or provide the following: {', '.join(missing_criteria)}."
