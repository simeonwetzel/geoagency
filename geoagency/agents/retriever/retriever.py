import asyncio
import aiohttp
import requests
from loguru import logger
from flashrank import Ranker, RerankRequest
import time
from ...config.config import CONFIG



class RepoRetriever:
    def __init__(self):
        self.repos = {}
        self.reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/opt")
        
        repos_config = CONFIG.repos.get("repositories")
        logger.info("Retriever initialized")

        # Add repositories to retriever
        for repo_name, repo_config in repos_config.items():
            logger.info(f"Adding repo to retriever: {repo_name}")
            self.add_repo(
                name=repo_name,
                base_url=repo_config.get("base_url"),
                params=repo_config.get("params"),
                headers=repo_config.get("headers"),
                response_keys=repo_config.get("response_keys")
            )


    def add_repo(self,
                 name: str,
                 base_url: str,
                 params: dict = None,
                 headers: dict = None,
                 response_keys: list = None) -> None:
        """
        Add a repository configuration.

        :param name: Name of the repository.
        :param base_url: Base URL of the repository API.
        :param params: Default query parameters for the repository API.
        :param headers: Default headers for the repository API.
        :param response_keys: Keys to extract from the response JSON.
        """
        self.repos[name] = {
            "base_url": base_url,
            "params": params or {},
            "headers": headers or {},
            "response_keys": response_keys or {},
        }

    def health_check(self) -> dict:
        """
        Perform a health check on all repositories.

        :return: A dictionary of health check results for each repository.
        """
        health_check_results = {}
        for repo_name in self.repos.keys():
            logger.info(f"Performing health check for repository: {repo_name}")
            health_check_results[repo_name] = self.health_check_repo(repo_name)
        return health_check_results

    def health_check_repo(self, repo_name: str) -> dict:
        """
        Perform a health check on a single repository.

        :param repo_name: Name of the repository.

        :return: A dictionary containing the health check result.
        """
        repo = self.repos.get(repo_name)
        if not repo:
            return {"error": f"Repository '{repo_name}' not found."}

        try:
            params = {**repo["params"], "q": "test"}
            response = requests.get(
                repo["base_url"], params=params, headers=repo["headers"])
            response.raise_for_status()
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def fetch_metadata(self,
                             session: aiohttp.ClientSession,
                             repo_name: str,
                             endpoint: str = "",
                             params: str = None) -> dict:
        """
        Fetch metadata from a single repository asynchronously.

        :param session: The aiohttp session object.
        :param repo_name: Name of the repository.
        :param endpoint: API endpoint for the repository.
        :param params: Query parameters for the request.

        :return: Response JSON or an error message.
        """
        repo = self.repos.get(repo_name)
        if not repo:
            return {"error": f"Repository '{repo_name}' not found."}

        url = f"{repo['base_url']}/{endpoint}".rstrip("/")
        all_params = {**repo["params"], **(params or {})}

        try:
            async with session.get(url, params=all_params, headers=repo["headers"]) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            return {"error": str(e)}

    async def retrieve_async(self, requests: list) -> dict:
        """
        Retrieve metadata from multiple repositories asynchronously.

        :param requests: A list of dictionaries specifying the repo_name, endpoint, and params.
        :return: A dictionary of responses for each request.
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for req in requests:
                repo_name = req["repo_name"]
                query = req.get("params", {}).get("q", "default_query")
                unique_key = f"{repo_name}:{query}"  # Create a unique key
                task = self.fetch_metadata(session, repo_name, req.get(
                    "endpoint", ""), req.get("params"))
                tasks.append((unique_key, task))

            results = await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
            return {key: result for (key, _), result in zip(tasks, results)}

    def format_response(self, response: dict, repo_name: str) -> dict:
        """
        Format the response JSON for a repository.

        :param response: Response JSON from the repository.
        :param repo_name: Name of the repository.

        :return: A list of dictionaries containing the formatted results.
        """
        results = [
            {
                'id': item.get('id') or item.get('metadata', {}).get('doi', ''),
                'source': repo_name,
                # 'title': item.get('title') or item.get('metadata', {}).get('title', ''),
                # 'description': item.get('description') or item.get('metadata', {}).get('description', ''),
                'text': f"{item.get('title') or item.get('metadata', {}).get('title', '')} - "
                        f"{item.get('description') or item.get('metadata', {}).get('description', '')}".strip(" -"),
                'meta': item
            }
            for item in response
        ]
        return results
        
    def rerank_results(self, query: str, results: list) -> list:
        """
        Rerank the results using a reranker model.

        :param query: Query string.
        :param results: A list of dictionaries containing the search results.

        :return: A list of dictionaries containing the reranked results.
        """
        start_time = time.perf_counter()
        rerank_request = RerankRequest(
            query=query,
            passages=results
        )
        reranked_results = self.reranker.rerank(rerank_request)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f"Time taken to re-rank {len(reranked_results)} records: {elapsed_time:.2f} seconds")
        return reranked_results
    
    # Define query and requests
    async def query_all_repos(self, q: str) -> dict:
        """
        Retrieve metadata for a query from all repositories asynchronously.

        :param q: Query string.
        :return: A dictionary containing the responses for each repository.
        """
        requests = [{"repo_name": repo_name, "params": {"q": q}}
                    for repo_name in self.repos.keys()]
        logger.info(f"Start retrieving metadata for query: {q}")
        start_time = time.perf_counter()

        responses = await self.retrieve_async(requests)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f"Time taken to retrieve metadata: {elapsed_time:.2f} seconds")

        # Process responses
        response_all_repos = []
        for request, response in responses.items():
            # Extract repo name and query from the request key
            repo_name, query = request.split(":")
            logger.info(f"Processing response for repo: {repo_name}, query: {query}")

            # Get the response keys (path) for the repo
            response_keys = self.repos.get(repo_name, {}).get(
                "response_keys", {}).get("path", [])

            # Traverse the nested dictionary using the response keys
            extracted_data = response
            for key in response_keys:
                if isinstance(extracted_data, dict) and key in extracted_data:
                    extracted_data = extracted_data[key]
                else:
                    logger.warning(
                        f"Key '{key}' not found in response for {repo_name}")
                    extracted_data = None
                    break

            # Log the count of results if data was successfully extracted
            if extracted_data is not None:
                count = len(extracted_data) if hasattr(
                    extracted_data, "__len__") else "unknown"
                response_all_repos += self.format_response(extracted_data, repo_name)
                logger.info(f"Repo: {repo_name}, Query: {query}, Retrieved results: {count}")
                
                # logger.info(
                #    f"---Result_titles: {[r.get('title', '') for r in extracted_data]}")
                
            else:
                logger.error(f"Could not extract data for {repo_name}, response keys path: {response_keys}")

        reranked_results = self.rerank_results(query=q, results=response_all_repos)

        # Convert scores from numpy floats to normal floats
        for result in reranked_results:
            if 'score' in result:
                result['score'] = float(result['score'])

        return {'query': q, 
                'results': reranked_results}
