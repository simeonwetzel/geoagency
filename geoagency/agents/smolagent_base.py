from smolagents import CodeAgent, ToolCallingAgent, ManagedAgent
import os
from ..llm_manager import LLMManager
from .tools.tools import MetadataRetrieverTool, clarify_search_criteria
from .retriever.retriever import RepoRetriever
from smolagents.prompts import MANAGED_AGENT_PROMPT

# Specify the model to be used 
llm = LLMManager.get_llm()

# Create different agents
retriever = RepoRetriever()
metadata_retriever_agent = CodeAgent(
    tools=[clarify_search_criteria, MetadataRetrieverTool(retriever)],
    model=llm, 
    additional_authorized_imports=["asyncio"],
    max_steps=1
)

# # Make managed agents that will be used by the manager agent
managed_metadata_agent = ManagedAgent(
     agent=metadata_retriever_agent,
     name="metadata_retriever",
     description="Retrieves metadata from different repositories and outputs the top-5 results. For any query about data or metadata use this agent.",
     managed_agent_prompt=MANAGED_AGENT_PROMPT + """Always use the query as is, do not modify it or hallucinate search criteria.""",
     #additional_prompting
)

manager_agent = CodeAgent(
    tools=[],
    model=llm,
    managed_agents=[managed_metadata_agent],
    max_steps=1
)
manager_agent.system_prompt = manager_agent.system_prompt + """
For each dataset/metadata query do a single request and show the results.
"""

def call_agent(query: str) -> str:
    return manager_agent.run(query)
