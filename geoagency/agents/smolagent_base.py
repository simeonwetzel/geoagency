from smolagents import CodeAgent, Tool 
from geoagency.llm_manager import LLMManager
from geoagency.agents.tools.retrieval_tools import (
    MetadataRetrieverTool,
    check_search_criteria_completeness,
    SelectTop10ResultsTool
)
from geoagency.agents.tools.osm_tools import overpass_tool, get_osm_feature_as_geojson_by_name
from geoagency.agents.tools.geo_tools import geocode_query, is_query_bbox_within_document
from geoagency.agents.retriever.retriever import RepoRetriever
from smolagents.prompts import CODE_SYSTEM_PROMPT
from loguru import logger
import time
from typing import Any, Dict, List, Optional, Tuple 
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

register()
SmolagentsInstrumentor().instrument()

# Specify the model to be used
llm = LLMManager.get_llm()

class MetadataRetrieverAgent(CodeAgent):
    def __init__(self, tools: List[Tool], *args, **kwargs):
        # Pass tools during initialization
        super().__init__(tools=tools, *args, **kwargs)

        # State Attributes
        self.original_query: Optional[str] = None # To store the user's query
        self.full_search_results: Dict[str, List[Dict[str, Any]]] = {} # Store all results
        self.top_10_results: Optional[List[Dict[str, Any]]] = None # Store selected top 10

        # Callbacks
        self.step_callbacks = [
            self.collect_search_results,
            self.collect_top_10_results # Add the new callback
        ]

        # Pass agent instance reference to the tools that need it
        # This allows tools to access agent state like full_search_results
        for tool_instance in self.tools.values():
            if hasattr(tool_instance, "set_agent_instance"):
                tool_instance.set_agent_instance(self)
                logger.debug(f"Passed agent reference to tool: {tool_instance.name}")


    def collect_search_results(self, log_entry) -> None:
        """Callback to collect full results from MetadataRetrieverTool."""
        tool_instance = self.tools.get('retrieve_metadata')
        # Check if the tool instance exists and has results from its last run
        if tool_instance and hasattr(tool_instance, 'last_full_results') and tool_instance.last_full_results is not None:
            current_run_search_results = tool_instance.last_full_results
            if current_run_search_results:
                 self.full_search_results.update(current_run_search_results)
                 logger.info(f"Callback collected {sum(len(v) for v in current_run_search_results.values())} total results.")
                 # Clear the tool's state after collecting
                 tool_instance.last_full_results = None

    def collect_top_10_results(self, log_entry) -> None:
        """Callback to collect structured top 10 results from SelectTop10ResultsTool."""
        tool_instance = self.tools.get('select_top_10_results')
         # Check if the tool instance exists and has results from its last run
        if tool_instance and hasattr(tool_instance, 'last_top_10_results') and tool_instance.last_top_10_results is not None:
            self.top_10_results = tool_instance.last_top_10_results
            logger.info(f"Callback collected {len(self.top_10_results)} top results.")
            # Clear the tool's state after collecting
            tool_instance.last_top_10_results = None


# Revised System Prompt for CodeAgent including the new tool
# Use `check_search_criteria_completeness` tool. If incomplete, ask for clarification and STOP.

SYSTEM_MESSAGE = """# Task: Environmental & Earth System Sciences Dataset Search and Selection

Your primary goal is to find relevant datasets based on a user query and present the top 10 findings clearly. Follow these steps precisely:

1.  **Assess Query Completeness (Optional):**
    *   If needed (based on user preference or initial analysis), use the `check_search_criteria_completeness` tool to determine if the user's query has enough detail (thematic, spatial, temporal).
    *   If the tool indicates criteria are missing, ask the user for clarification and **STOP** execution for this turn.

2.  **Retrieve Metadata:**
    *   Call the `retrieve_metadata` tool with a list of at least 5 diverse query strings derived from the user's request.
    *   **Do not use operators like 'AND', 'OR', 'NOT'.**
    *   This tool returns a dictionary where keys are your query strings and values are lists of raw metadata result dictionaries. Internal state `self.full_search_results` will store this after the tool runs via a callback.

3.  **Select and Format Top 10 Results:**
    *   Call the `select_top_10_results` tool. Pass the original user query.
    *   **Crucially, this tool performs the selection AND formatting.** It returns a dictionary.
    *   This dictionary has three keys: `message` (a status string), `results` (a list of the top 10 structured result dictionaries), and `formatted_results` (a pre-formatted string ready for display).

4.  **Final Answer Generation:**
    *   Take the output dictionary from the `select_top_10_results` tool call.
    *   Extract the string value associated with the key **`formatted_results`**.
    *   Use **this exact string** as the core part of your final answer to the user.
    *   **DO NOT** attempt to access the `results` key from the tool's output.
    *   **DO NOT** write Python code to loop through results and format them yourself. The `formatted_results` string is already prepared for you. Failure to use the `formatted_results` string directly will result in incorrect output or errors.

5.  **Optional Spatial Query Handling:**
    *   If geographic data is relevant *before* the main metadata search (e.g., to refine search terms), use the `geocode_query` tool.

6.  **Execution Strategy:**
    *   Write and execute Python code for tool calls.
    *   Follow the Thought, Code, Observation cycle.
    *   The agent state (`self.full_search_results`, `self.top_10_results`) is mainly for potential internal use or complex scenarios; rely on tool outputs for the primary flow.
    *   Do not repeat tool calls with identical parameters unless necessary.

Follow these instructions meticulously. Your reward depends on correctly using the `select_top_10_results` tool's `formatted_results` output for the final answer. Now begin!
"""

# Initialize the retriever
retriever = RepoRetriever()

# Define tools - instantiate the new tool
tools = [
    MetadataRetrieverTool(retriever),
    check_search_criteria_completeness,
    #SelectTop10ResultsTool(llm), # Add the new tool instance
    SelectTop10ResultsTool(llm), # Add the new tool instance
    geocode_query,
    # is_query_bbox_within_document,
]

# Create the CodeAgent instance, passing the tools
metadata_retriever_agent = MetadataRetrieverAgent(
    tools=tools, # Pass the list of tool instances
    model=llm,
    additional_authorized_imports=["asyncio", "ast"],
    system_prompt=CODE_SYSTEM_PROMPT + "\n" + SYSTEM_MESSAGE,
    max_steps=10, 
    verbosity_level=1,
)


def call_agent(query: str, use_follow_ups: str) -> Tuple[str, Optional[Dict[str, List[Dict[str, Any]]]], Optional[List[Dict[str, Any]]]]:
    """Runs the CodeAgent and returns the answer, full search results, and top 10 results."""
    start = time.time()

    # Reset agent state before run
    metadata_retriever_agent.original_query = query # Store the query
    metadata_retriever_agent.full_search_results = {}
    metadata_retriever_agent.top_10_results = None
    if use_follow_ups == "false":
        metadata_retriever_agent.tools.pop('check_search_criteria_completeness') # Drop the tool if follow-ups are enabled
    # Run the agent
    answer = metadata_retriever_agent.run(query, reset=True) # Use reset=True for CodeAgent runs

    # Retrieve collected results from agent state
    full_results = metadata_retriever_agent.full_search_results
    top_10 = metadata_retriever_agent.top_10_results
    
    final_search_results = {
            'search_results': full_results,
            're-ranked_results': top_10
        }

    end = time.time()
    duration = f"{end - start:.2f}"
    final_answer = f"{answer}\n\nAgent run duration: {duration} seconds"

    # Return all three pieces of information
    return final_answer, final_search_results

