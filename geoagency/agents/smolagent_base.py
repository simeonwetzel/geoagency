from smolagents import CodeAgent, ToolCallingAgent, ManagedAgent
import os
from geoagency.llm_manager import LLMManager
from geoagency.agents.tools.retrieval_tools import MetadataRetrieverTool, clarify_search_criteria
from geoagency.agents.tools.osm_tools import overpass_tool, get_osm_feature_as_geojson_by_name
from geoagency.agents.tools.geo_tools import geocode_query
from geoagency.agents.retriever.retriever import RepoRetriever
from smolagents.prompts import MANAGED_AGENT_PROMPT, CODE_SYSTEM_PROMPT 
from loguru import logger
import ast 
import re 

# Specify the model to be used 
llm = LLMManager.get_llm()

# Create different agents
search_results = []

class MetadataRetrieverAgent(CodeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_results = []  # Initialize an attribute to store search results
        self.step_callbacks = [self.collect_search_results]
    
    def collect_search_results(self, log_entry) -> None:
        """
        This function writes the structured output of a RepoRetriever retriever to the search_results attribute of this agent.
        We do this, because we want to also have a machine-readable object of our search results. 
        """       

        if log_entry.observations and 'retrieve_metadata' in log_entry.tool_calls[0].arguments:
            match = re.search(r"Execution logs:\n(.*)\nLast output from code snippet:", log_entry.observations, re.DOTALL)
            if match:
                try:
                    log_content = match.group(1)
                    log_dict = ast.literal_eval(log_content)
                    structured_data = log_dict.get('structured_data', [])
                    self.search_results.extend(structured_data)
                    logger.info(structured_data)  # Output the parsed structured data
                except (SyntaxError, ValueError) as e:
                    logger.warning(f"Error parsing structured data: {e}")
            else:
                logger.info("No structured data found.")
                    
retriever = RepoRetriever()  
metadata_retriever_agent = MetadataRetrieverAgent(
    tools=[clarify_search_criteria, MetadataRetrieverTool(retriever), geocode_query],
    model=llm, 
    additional_authorized_imports=["asyncio"],
    system_prompt=f"""Your task is to retrieve metadata. Only do a one-shot search (not multiple times)!\n {CODE_SYSTEM_PROMPT}""",
    max_steps=3,
    verbosity_level=1,
)


def call_agent(query: str) -> str:
    # return manager_agent.run(query)   
    answer = metadata_retriever_agent.run(query, reset=True)
    search_results = metadata_retriever_agent.search_results
    return answer, search_results