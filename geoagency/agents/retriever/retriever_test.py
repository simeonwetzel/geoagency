from .retriever import RepoRetriever
import asyncio
from loguru import logger
from geoagency.config.config import CONFIG
import time
import nest_asyncio

nest_asyncio.apply()

# Initialize retriever and configuration
retriever = RepoRetriever()
# repos_config = CONFIG.repos.get("repositories")
logger.info("Retriever initialized")


list_queries = ["precipitation"]

results = asyncio.run(retriever.query_multiple_repos(queries=list_queries, limit=5))

print(results)