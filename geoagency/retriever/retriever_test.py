from retriever import RepoRetriever
import asyncio
from loguru import logger
from geoagency.config.config import CONFIG
import time
import nest_asyncio

nest_asyncio.apply()

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

# Define query and requests
async def query_all_repos(q: str) -> dict:
    """
    Retrieve metadata for a query from all repositories asynchronously.
    
    :param q: Query string.
    :return: A dictionary containing the responses for each repository.
    """
    requests = [{"repo_name": repo_name, "params": {"q": q}} for repo_name in retriever.repos.keys()]
    logger.info(f"Start retrieving metadata for query: {q}")
    start_time = time.perf_counter()
    
    responses = await retriever.retrieve_async(requests)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.info(f"Time taken to retrieve metadata: {elapsed_time:.2f} seconds")

    # Process responses
    for request, response in responses.items():
        repo_name, query = request.split(":")  # Extract repo name and query from the request key
        logger.info(f"Processing response for repo: {repo_name}, query: {query}")

        # Get the response keys (path) for the repo
        response_keys = retriever.repos.get(repo_name, {}).get("response_keys", {}).get("path", [])
        
        # Traverse the nested dictionary using the response keys
        extracted_data = response
        for key in response_keys:
            if isinstance(extracted_data, dict) and key in extracted_data:
                extracted_data = extracted_data[key]
            else:
                logger.warning(f"Key '{key}' not found in response for {repo_name}")
                extracted_data = None
                break
        
        # Log the count of results if data was successfully extracted
        if extracted_data is not None:
            count = len(extracted_data) if hasattr(extracted_data, "__len__") else "unknown"
            logger.info(f"Repo: {repo_name}, Query: {query}, Retrieved results: {count}")
            logger.info(f"---Result_titles: {[r.get('title', '') for r in extracted_data]}")
        else:
            logger.error(f"Could not extract data for {repo_name}, response keys path: {response_keys}")

    return response

# Define the query string
response = asyncio.run(query_all_repos(q="climate change"))
logger.info(response)