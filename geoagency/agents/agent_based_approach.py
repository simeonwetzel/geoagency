import json
from geoagency.agents.retriever.retriever import RepoRetriever
from smolagents import CodeAgent, Tool, tool
from loguru import logger
import time
from typing import Any, Dict, List, Optional, Tuple
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from geoagency.agents.tools.retrieval_tools import MetadataRetrieverTool, ResultReRankingTool
from smolagents.agents import ActionStep 
import re
from geoagency.llm_manager import LLMManager
from geoagency.agents.tools.geo_tools import geocode_query

# Specify the model to be used
llm = LLMManager.get_llm()

class MetadataRetrieverAgent(CodeAgent):
    def __init__(self, tools: List[Tool], *args, **kwargs):
        # Pass tools during initialization
        super().__init__(tools=tools, *args, **kwargs)

        # State Attributes
        self.original_query: Optional[str] = None  # To store the user's query
        # Store all results
        self.full_search_results: Dict[str, List[Dict[str, Any]]] = {}
        self.query_results = []  # Flattened and deduplicated results
        self.ranked_results = []  # Store the top ranked results
        self.query_criteria = {}  # Store the query criteria

        # Callbacks
        self.step_callbacks = [
            self._truncate_memory_,
        ]
        # Set up finished flag to avoid repeated access errors
        self._setup_completed = False
        self.setup_tools()


    def setup_tools(self):
        """Set up tool references after initialization."""
        if self._setup_completed:
            return

        # Pass agent instance reference to the tools that need it
        # This allows tools to access agent state like full_search_results
        for tool_name, tool_instance in self.tools.items():
            if hasattr(tool_instance, "agent_instance"):
                tool_instance.agent_instance = self
                logger.debug(f"Passed agent reference to tool: {tool_name}")

        self._setup_completed = True
        
    def _truncate_memory_(self, memory_step: ActionStep) -> ActionStep:        
        current_step = memory_step.step_number
        
        if hasattr(memory_step, 'model_input_messages'):
            for message in memory_step.model_input_messages:
                if len(message.get('content', [])[0].get('text')) > 1000:
                    logger.warning(f"Truncating input_messages for step {current_step}")
                    message['content'][0]['text'] = f"{message['content'][0]['text'][:1000]}..."    
                    
        
        if hasattr(memory_step, 'observations') and memory_step.observations is not None:
            if len(memory_step.observations) > 1000:
                logger.warning(f"Truncating input_messages for step {current_step}")
                memory_step.observations = memory_step.observations[:1000]

        return

retriever = RepoRetriever()

# Agent configuration
tools = [
    MetadataRetrieverTool(retriever),
    # DecomposeQueryTool(llm),
    ResultReRankingTool(llm),
    # geocode_query
    # understand_query
]

SYSTEM_PROMPT = """Task: you need to assist users in finding metadata for environmental datasets.
You are able to search for data and re-rank the results and then present the user your findings.

Alwyays first use the retriever and then the re-ranker.

Do not try to analyze the found data behind the metadata. Just present the top hits as a metadata catalogue would do. 
"""

metadata_retriever_agent = MetadataRetrieverAgent(
    tools=tools,  # Pass the list of tool instances
    model=llm,
    additional_authorized_imports=["asyncio", "ast", "requests"],
    max_steps=12,
    verbosity_level=2,
)

metadata_retriever_agent.prompt_templates["system_prompt"] = metadata_retriever_agent.prompt_templates["system_prompt"] + f"\n{SYSTEM_PROMPT}"


def call_agent(query: str, use_follow_ups: bool = False) -> Tuple[str, Optional[Dict[str, List[Dict[str, Any]]]], Optional[List[Dict[str, Any]]]]:
    """Runs the CodeAgent and returns the answer, full search results, and top 10 results."""
    start = time.time()

    # Reset agent state before run
    metadata_retriever_agent.original_query = query  # Store the query
    metadata_retriever_agent.last_full_results = []
    metadata_retriever_agent.ranked_results = []  # Reset ranked results
    metadata_retriever_agent._setup_completed = False  # Reset setup flag

    # Make sure tools are set up properly
    metadata_retriever_agent.setup_tools()

    try:
        # Set up a backoff strategy for rate limit errors
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Run the agent
                # Use reset=True for CodeAgent runs
                prompt = f"{query}"
                answer = metadata_retriever_agent.run(prompt, reset=True)
                break  # If successful, exit the retry loop
            except Exception as e:
                if "rate_limit_exceeded" in str(e) and retry_count < max_retries - 1:
                    retry_count += 1
                    logger.warning(
                        f"Rate limit exceeded. Retrying in 5 seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(5)  # Wait before retrying
                else:
                    # If it's not a rate limit error or we've exhausted retries, raise the exception
                    raise
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        answer = f"An error occurred while processing your query: {str(e)}"

    end = time.time()
    duration = f"{end - start:.2f}"
    final_answer = f"{answer}\n\nAgent run duration: {duration} seconds"

    # Return answer, full search results, and top ranked results (or all results if not ranked)
    full_results = metadata_retriever_agent.last_full_results
    top_results = metadata_retriever_agent.ranked_results if metadata_retriever_agent.ranked_results else full_results[
        :10]
    
    
    final_search_results = {
            'search_results': full_results,
            're-ranked_results': top_results
        }

    return final_answer, final_search_results
