from smolagents import CodeAgent, HfApiModel
import os
from ..llm_manager import LLMManager
from .tools.tools import retrieve_metadata, format_metadata_results
from .retriever.retriever import RepoRetriever


llm = LLMManager.get_llm()
agent = CodeAgent(tools=[retrieve_metadata, format_metadata_results], model=llm, additional_authorized_imports=["asyncio"])

def call_agent(query: str, retriever: RepoRetriever) -> str:
    return agent.run(query,
                     additional_args={"retriever": retriever})
