import asyncio
import aiohttp
import requests
from loguru import logger
from flashrank import Ranker, RerankRequest
import time
import shapely
from ...config.config import CONFIG
from typing import Any


def get_nested_value(d, keys, default=None):
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default  # Return a default if key is missing
    return d


class RepoRetriever:
    def __init__(self):
        self.repos = {}
        self.reranker = Ranker(
            model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/opt")

        repos_config = CONFIG.repos.get("repositories")
        logger.info("Retriever initialized")

        # Add repositories to retriever
        for repo_name, repo_config in repos_config.items():
            logger.info(f"Adding repo to retriever: {repo_name}")
            params = repo_config.get("params")
            query_key = next(iter({"q", "query"} & params.keys()), None)

            params[query_key] = "test"
            self.add_repo(
                name=repo_name,
                base_url=repo_config.get("base_url"),
                query_key=query_key,
                params=params,
                headers=repo_config.get("headers"),
                response_keys=repo_config.get("response_keys"),
                geometry_field=repo_config.get("geometry_field"),
                source_url_field=repo_config.get("source_url_field"),
            )

    def add_repo(self,
                 name: str,
                 base_url: str,
                 query_key: str = None,
                 params: dict = None,
                 headers: dict = None,
                 response_keys: list = None,
                 geometry_field: list = None,
                 source_url_field: str = None) -> None:
        """
        Add a repository configuration.

        :param name: Name of the repository.
        :param base_url: Base URL of the repository API.
        :param params: Default query parameters for the repository API.
        :param headers: Default headers for the repository API.
        :param response_keys: Keys to extract from the response JSON.
        :param geometry_field: Name of the geoemetry field in the response JSON.
        """
        self.repos[name] = {
            "base_url": base_url,
            "query_key": query_key or 'q',
            "params": params or {},
            "headers": headers or {},
            "response_keys": response_keys or {},
            "geometry_field_path": geometry_field or {},
            "source_url_field": source_url_field or "",
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
            params = {**repo["params"]}
            query_key = params.get("query_key")
            params[query_key] = "test"
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
                query_key = self.repos[repo_name]["query_key"]
                query = req.get("params", {}).get(query_key, "default_query")
                unique_key = f"{repo_name}:{query}"  # Create a unique key
                logger.info(f"Query: {unique_key}")

                # Ensure params is a dictionary or a list of key-value pairs
                params = req.get("params", {})
                if not isinstance(params, dict):
                    logger.error(f"Invalid params for {unique_key}: {params}")
                    params = {}  # Default to empty dict if invalid

                task = self.fetch_metadata(session, repo_name, req.get(
                    "endpoint", ""), req.get("params"))
                tasks.append((unique_key, task))

            results = await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
            return {key: result for (key, _), result in zip(tasks, results)}

    async def query_multiple_repos(self, queries: list, limit: int=None) -> dict:
        """
        Retrieve metadata for multiple queries from all repositories asynchronously.

        :param queries: List of query strings.
        :param limit: Limit the number of results returned per query
        :return: A dictionary containing results grouped by query.
        """
        logger.info(f"Start retrieving metadata for queries: {queries}")

        start_time = time.perf_counter()

        # Construct requests for all queries across all repositories
        requests = [
            {"repo_name": repo_name, "params": {**self.repos[repo_name]["params"], self.repos[repo_name]["query_key"]: q}}
            for q in queries for repo_name in self.repos
        ]

        # Fetch all responses concurrently
        responses = await self.retrieve_async(requests)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f"Time taken to retrieve metadata: {elapsed_time:.2f} seconds")

        # Process responses and organize results by query
        query_results = {q: {} for q in queries}

        for request_key, response in responses.items():
            repo_name, query = request_key.split(":")
            response_keys = self.repos[repo_name].get("response_keys", {}).get("path", [])

            extracted_data = response
            for key in response_keys:
                if isinstance(extracted_data, dict) and key in extracted_data:
                    extracted_data = extracted_data[key]
                else:
                    logger.warning(f"Key '{key}' not found in response for {repo_name}")
                    extracted_data = None
                    break

            if extracted_data is not None:
                query_results[query][repo_name] = self.format_response(extracted_data, repo_name)
                logger.info(f"Repo: {repo_name}, Query: {query}, Retrieved {len(extracted_data)} results")

            else:
                query_results[query][repo_name] = []
                logger.error(f"Could not extract data for {repo_name}, response keys path: {response_keys}")

        # Rerank results per query
        for query in queries:
            all_results = [result for repo_results in query_results[query].values() for result in repo_results]
            reranked_results = self.rerank_results(query, all_results)
            for result in reranked_results:
                if 'score' in result:
                    result['score'] = float(result['score'])
            if limit:
                reranked_results_cutoff = reranked_results[:limit]
                query_results[query] = reranked_results_cutoff
            else:
                query_results[query] = reranked_results

        return query_results


    def _geometry_to_geojson(self, geometry: Any) -> dict:
        if geometry:
            if isinstance(geometry, str):
                geometry = shapely.from_wkt(geometry=geometry)
                return shapely.to_geojson(geometry=geometry)
            if isinstance(geometry, list):
                geometry = geometry[0]
            if isinstance(geometry, dict):
                if 'min' in geometry and 'max' in geometry:
                    # Assuming geometry is a bounding box like
                    # {"min": {"x": -17.15, "y": 36.93}, "max": {"x": 42.95, "y": 57.73}}
                    polygon = shapely.Polygon([
                        # Bottom-left
                        (geometry["min"]["x"], geometry["min"]["y"]),
                        # Bottom-right
                        (geometry["max"]["x"], geometry["min"]["y"]),
                        # Top-right
                        (geometry["max"]["x"], geometry["max"]["y"]),
                        (geometry["min"]["x"],
                         geometry["max"]["y"]),  # Top-left
                        # Close the polygon
                        (geometry["min"]["x"], geometry["min"]["y"])
                    ])
                    return shapely.to_geojson(geometry=polygon)
            return ''

    def format_response(self, response: dict, repo_name: str) -> dict:
        """
        Format the response JSON for a repository.

        :param response: Response JSON from the repository.
        :param repo_name: Name of the repository.

        :return: A list of dictionaries containing the formatted results.
        """
        # Get the keys of the geometry field within the result object
        geometry_field_path = self.repos.get(repo_name, {}).get(
            "geometry_field_path", {}).get("path", [])
        
        source_url_field = self.repos.get(repo_name, {}).get(
            "source_url_field", "")
         

        logger.info(f"Geometry field path is {geometry_field_path}")
        logger.info(f"Source URL field is: {source_url_field}")

        results = [
            {
                'id': item.get('id') or item.get('metadata', {}).get('doi', ''),
                'source': item.get(source_url_field) or repo_name,
                # 'title': item.get('title') or item.get('metadata', {}).get('title', ''),
                # 'description': item.get('description') or item.get('metadata', {}).get('description', ''),
                'text': f"{item.get('title') or item.get('metadata', {}).get('title', '')} - "
                        f"{item.get('description') or item.get('metadata', {}).get('description', '')}".strip(
                            " -"),
                'geometry_geojson': self._geometry_to_geojson(get_nested_value(item, geometry_field_path))
                if geometry_field_path else "" or "",
                'meta': item,
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
        logger.info(
            f"Time taken to re-rank {len(reranked_results)} records: {elapsed_time:.2f} seconds")
        return reranked_results

    # Define query and requests
    async def query_all_repos(self, q: str) -> dict:
        """
        Retrieve metadata for a query from all repositories asynchronously.

        :param q: Query string.
        :return: A dictionary containing the responses for each repository.
        """
        for repo_name in self.repos:
            query_key = self.repos[repo_name]["query_key"]
            self.repos[repo_name]["params"][query_key] = q

        requests = [{"repo_name": repo_name,
                     "params": self.repos[repo_name]["params"]}
                    for repo_name in self.repos.keys()]
        logger.info(f"Start retrieving metadata for query: {q}")

        start_time = time.perf_counter()

        responses = await self.retrieve_async(requests)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(
            f"Time taken to retrieve metadata: {elapsed_time:.2f} seconds")

        # Process responses
        response_all_repos = []
        for request, response in responses.items():
            # Extract repo name and query from the request key
            repo_name, query = request.split(":")
            logger.info(
                f"Processing response for repo: {repo_name}, query: {query}")

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
                response_all_repos += self.format_response(
                    extracted_data, repo_name)
                logger.info(
                    f"Repo: {repo_name}, Query: {query}, Retrieved results: {count}")

                # logger.info(
                #    f"---Result_titles: {[r.get('title', '') for r in extracted_data]}")

            else:
                logger.error(
                    f"Could not extract data for {repo_name}, response keys path: {response_keys}")

        reranked_results = self.rerank_results(
            query=q, results=response_all_repos)

        # Convert scores from numpy floats to normal floats
        for result in reranked_results:
            if 'score' in result:
                result['score'] = float(result['score'])

        return {'query': q,
                'results': reranked_results}
