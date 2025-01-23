from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger

from .config.config import (
    CONFIG,
    resolve_abs_path
)

from .agents.smolagent_base import call_agent

from .llm_manager import LLMManager

llm = LLMManager.get_llm()

from .agents.retriever.retriever import RepoRetriever

# Initialize retriever and configuration
retriever = RepoRetriever()
repos_config = CONFIG.repos.get("repositories")
logger.info("Retriever initialized")

# Add repositories to retriever
for repo_name, repo_config in repos_config.items():
    logger.info(f"Adding repo to retriever: {repo_name}")
    retriever.add_repo(
        name=repo_name,
        base_url=repo_config.get("base_url"),
        params=repo_config.get("params"),
        headers=repo_config.get("headers"),
        response_keys=repo_config.get("response_keys")
    )

# Perform health check on all repositories
health_check_results = retriever.health_check()
logger.info(f"Health check results: {health_check_results}")

for repo_name, health_status in health_check_results.items():
    if health_status.get("status") != "healthy":
        retriever.repos.pop(repo_name)
        logger.info(f"Repository '{repo_name}' unattached from retriever as connection unhealthy.")

app = FastAPI()


@app.get("/test_llm")
def read_root(query: str) -> dict:
    return {query: call_agent(query, retriever)}

@app.get("/retrieve_metadata")
async def retrieve_metadata(query: str) -> dict:
    response = await(retriever.query_all_repos(query))
    return {"response": response}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}