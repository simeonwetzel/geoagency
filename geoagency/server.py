from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from typing import List

from .config.config import (
    CONFIG,
    resolve_abs_path
)

from .agents.smolagent_base import call_agent
# from .agents.agent_factory import call_agent as call_agency
from .agents.deterministic_approach import call_agent as call_deterministic

from .llm_manager import LLMManager

llm = LLMManager.get_llm()

from .agents.retriever.retriever import RepoRetriever

# Initialize retriever and configuration
retriever = RepoRetriever()
repos_config = CONFIG.repos.get("repositories")
app = FastAPI()


@app.get("/test_llm")
def read_root(query: str) -> dict:
    answer, search_results = call_agent(query)
    # logger.debug(f"State: {logs}")
    return {"answer": answer, 
            "search_results": search_results}
    
@app.get("/test_deterministic")
async def read_root(query: str, use_follow_ups: str) -> dict:
    answer, search_results = await call_deterministic(query, use_follow_ups)
    # logger.debug(f"State: {logs}")
    return {"answer": answer, 
            "search_results": search_results}

"""
@app.get("/test_agent_factory")
def read_root(query: str) -> dict:
    answer = call_agency(query)
    # logger.debug(f"State: {logs}")
    return {"answer": answer}
"""
    
@app.get("/retrieve_metadata")
async def retrieve_metadata(query: str) -> dict:
    response = await(retriever.query_all_repos(query))
    return {"response": response}

@app.post("/retrieve_metadata_multi_query")
async def retrieve_metadata_multi_query(queries: List[str], limit: int=None) -> dict:
    response = await(retriever.query_multiple_repos(queries=queries, 
                                                    limit=limit))
    return {"response": response}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}