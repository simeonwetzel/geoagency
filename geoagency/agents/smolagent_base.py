from smolagents import CodeAgent, HfApiModel
import os
from ..llm_manager import LLMManager

llm = LLMManager.get_llm()
agent = CodeAgent(tools=[], model=llm)

def call_agent(query: str) -> str:
    return agent.run(query)
