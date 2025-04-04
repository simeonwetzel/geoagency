from smolagents import CodeAgent, Tool # Keep CodeAgent
import os
from geoagency.llm_manager import LLMManager
# Import the revised tools, including the new one
from geoagency.agents.tools.retrieval_tools import (
    MetadataRetrieverTool,
    check_search_criteria_completeness,
    SelectTop10ResultsTool # Import the new tool
)
from geoagency.agents.tools.osm_tools import overpass_tool, get_osm_feature_as_geojson_by_name
from geoagency.agents.tools.geo_tools import geocode_query, is_query_bbox_within_document
from geoagency.agents.retriever.retriever import RepoRetriever
from smolagents.prompts import CODE_SYSTEM_PROMPT
from loguru import logger
import re
import time
from typing import Any, Dict, List, Optional, Tuple # Ensure Tuple is imported

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

        # Pass agent instance reference TO the tools that need it
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
system_message = """# You are an expert assistant using Python code to search for Environmental and Earth System Sciences datasets.

# Workflow:
1.  **Understand & Check:** Analyze the user's query. Use `check_search_criteria_completeness` tool to ensure sufficient details (thematic, spatial, temporal if needed). If incomplete, ask the user for clarification and STOP.
2.  **Retrieve:** If criteria are complete, generate relevant query strings and use the `retrieve_metadata` tool to fetch results. You will receive a text summary.
3.  **Select Top 10:** AFTER `retrieve_metadata` succeeds, use the `select_top_10_results` tool. You MUST provide the original user query to this tool, like `select_top_10_results(original_query="<The original query from the user>")`. This tool selects the 10 most relevant results from the full set retrieved earlier. You will receive a confirmation message.
4.  **Synthesize Final Answer:** Formulate a concise, natural language final answer for the user. Base this answer PRIMARILY on the summary you received from `retrieve_metadata`. You can optionally mention that a top-10 list was also generated if the `select_top_10_results` tool succeeded. Do NOT just list the top 10 raw data in the final answer unless specifically asked. Focus on a helpful summary.

# Important:
- Write and execute Python code to call tools: `check_search_criteria_completeness`, `retrieve_metadata`, `select_top_10_results`.
- The `select_top_10_results` tool MUST be called AFTER `retrieve_metadata` and MUST include the `original_query` argument.
- Base your final natural language answer on the **summary** from `retrieve_metadata`, not the raw top-10 list.
"""

# Initialize the retriever
retriever = RepoRetriever()

# Define tools - instantiate the new tool
tools = [
    MetadataRetrieverTool(retriever),
    check_search_criteria_completeness,
    SelectTop10ResultsTool(), # Add the new tool instance
    geocode_query,
    # is_query_bbox_within_document,
    # DuckDuckGoSearchTool()
]

# Create the CodeAgent instance, passing the tools
metadata_retriever_agent = MetadataRetrieverAgent(
    tools=tools, # Pass the list of tool instances
    model=llm,
    additional_authorized_imports=["asyncio"],
    system_prompt=CODE_SYSTEM_PROMPT + "\n" + system_message,
    max_steps=10, # Increased slightly for the extra tool call step
    verbosity_level=1,
)

# Modified call_agent function
def call_agent(query: str) -> Tuple[str, Optional[Dict[str, List[Dict[str, Any]]]], Optional[List[Dict[str, Any]]]]:
    """Runs the CodeAgent and returns the answer, full search results, and top 10 results."""
    start = time.time()

    # Reset agent state before run
    metadata_retriever_agent.original_query = query # Store the query
    metadata_retriever_agent.full_search_results = {}
    metadata_retriever_agent.top_10_results = None

    # Run the agent
    answer = metadata_retriever_agent.run(query, reset=True) # Use reset=True for CodeAgent runs

    # Retrieve collected results from agent state
    full_results = metadata_retriever_agent.full_search_results
    top_10 = metadata_retriever_agent.top_10_results
    full_results['top_10_hits'] = top_10
    end = time.time()
    duration = f"{end - start:.2f}"
    final_answer = f"{answer}\n\nAgent run duration: {duration} seconds"

    # Return all three pieces of information
    return final_answer, full_results

