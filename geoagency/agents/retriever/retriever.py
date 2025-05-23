import asyncio
import aiohttp
import requests
from loguru import logger
from flashrank import Ranker, RerankRequest
import time
import shapely
from ...config.config import CONFIG
from typing import Any
import urllib
import json


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

            # Get basic parameters
            params = repo_config.get("params", {})
            query_format = repo_config.get("query_format", "standard")
            query_template = repo_config.get("query_template")

            # Determine query key based on the data available
            query_key = repo_config.get("query_key")
            if not query_key and query_format == "standard":
                query_key = next(iter({"q", "query"} & params.keys()), "q")
            elif not query_key:
                query_key = "q"  # Default fallback

            self.add_repo(
                name=repo_name,
                base_url=repo_config.get("base_url"),
                query_key=query_key,
                params=params,
                headers=repo_config.get("headers", {}),
                method=repo_config.get("method", "GET"),
                response_keys=repo_config.get("response_keys"),
                geometry_field=repo_config.get("geometry_field"),
                source_url_field=repo_config.get("source_url_field"),
                field_mapping=repo_config.get("field_mapping"),
                query_format=query_format,
                query_template=query_template,
            )

    def add_repo(self,
                 name: str,
                 base_url: str,
                 query_key: str = None,
                 params: dict = None,
                 headers: dict = None,
                 method: str = None,
                 response_keys: list = None,
                 geometry_field: list = None,
                 source_url_field: str = None,
                 field_mapping: dict = None,
                 query_format: str = None,
                 query_template: dict = None
                 ) -> None:
        """
        Add a repository configuration.
        """
        self.repos[name] = {
            "base_url": base_url,
            "query_key": query_key or 'q',
            "params": params or {},
            "headers": headers or {},
            "method": method or "GET",
            "response_keys": response_keys or {},
            "field_mapping": field_mapping or {},
            "geometry_field_path": geometry_field or {},
            "source_url_field": source_url_field or "",
            "query_format": query_format or "standard",
            "query_template": query_template
        }
        

    def construct_query_params(self, repo_name: str, query: str) -> dict:
        repo = self.repos.get(repo_name)
        if not repo:
            return {}

        params = {**repo.get("params", {})}
        query_format = repo.get("query_format", "standard")

        # Standard: simple param
        if query_format == "standard":
            params[repo.get("query_key", "q")] = query
            return params

        # JSON encoded: build and URL-encode
        if query_format == "json_encoded":
            template = repo.get("query_template", {})
            query_data = json.loads(json.dumps(template))  # deep copy

            def replace(d):
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(v, (dict, list)):
                            replace(v)
                        elif isinstance(v, str) and "{{query}}" in v:
                            d[k] = v.replace("{{query}}", query)
                elif isinstance(d, list):
                    for i, v2 in enumerate(d):
                        if isinstance(v2, (dict, list)):
                            replace(v2)
                        elif isinstance(v2, str) and "{{query}}" in v2:
                            d[i] = v2.replace("{{query}}", query)
            replace(query_data)
            return {"query": urllib.parse.quote(json.dumps(query_data))}

        # JSON template: return full body
        if query_format == "json_template":
            template = repo.get("query_template", {})
            body = json.loads(json.dumps(template))
            # replace placeholders

            def replace(d):
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(v, (dict, list)):
                            replace(v)
                        elif isinstance(v, str) and "{query}" in v:
                            d[k] = v.replace("{query}", query)
                elif isinstance(d, list):
                    for i, v2 in enumerate(d):
                        if isinstance(v2, (dict, list)):
                            replace(v2)
                        elif isinstance(v2, str) and "{query}" in v2:
                            d[i] = v2.replace("{query}", query)
            replace(body)
            return {"body": body}

        return params

    def health_check_repo(self, repo_name: str) -> dict:
        repo = self.repos.get(repo_name)
        if not repo:
            return {"error": f"Repository '{repo_name}' not found."}
        try:
            params = {**repo.get("params", {})}
            params[repo.get("query_key")] = "test"
            method = repo.get("method").upper()
            headers = repo.get("headers", {})
            if method == "GET":
                response = requests.get(
                    repo["base_url"], params=params, headers=headers)
            else:
                # handle json_template
                if repo.get("query_format") == "json_template":
                    body = self.construct_query_params(
                        repo_name, "test").get("body")
                    response = requests.post(
                        repo["base_url"], json=body, headers=headers)
                else:
                    response = requests.post(
                        repo["base_url"], json=params, headers=headers)
            response.raise_for_status()
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def fetch_metadata(self,
                            session: aiohttp.ClientSession,
                            repo_name: str,
                            endpoint: str = "",
                            params: dict = None) -> dict:
        repo = self.repos.get(repo_name)
        if not repo:
            return {"error": f"Repository '{repo_name}' not found."}

        url = f"{repo['base_url']}/{endpoint}".rstrip("/")
        headers = repo.get("headers", {})
        query_format = repo.get("query_format")
        query_key = repo.get("query_key", "q")
        
        # Get the query value from params
        query_value = params.get(query_key, "")
        
        # Construct the parameters with the query
        constructed = self.construct_query_params(repo_name, query_value)

        try:
            # GET requests
            if repo.get("method").upper() == "GET":
                return await (await session.get(url, params=constructed, headers=headers)).json()
            # POST requests
            if query_format == "json_template":
                body = constructed.get("body", {})
                async with session.post(url, json=body, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                async with session.post(url, json=constructed, headers=headers) as response:
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

    async def query_multiple_repos(self, queries: list, limit: int = None) -> dict:
        """
        Retrieve metadata for multiple queries from all repositories asynchronously.

        :param queries: List of query strings.
        :param limit: Limit the number of results returned per query
        :return: A dictionary containing results grouped by query.
        """
        logger.info(f"Start retrieving metadata for queries: {queries}")

        start_time = time.perf_counter()

        # Construct requests for all queries across all repositories
        requests = []
        for q in queries:
            for repo_name in self.repos:
                repo = self.repos[repo_name]
                query_key = repo.get("query_key", "q")
                # Only set the query key, let construct_query_params handle the rest
                params = {query_key: q}
                requests.append({"repo_name": repo_name, "params": params})

        # Fetch all responses concurrently
        responses = await self.retrieve_async(requests)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(
            f"Time taken to retrieve metadata: {elapsed_time:.2f} seconds")

        # Process responses and organize results by query
        query_results = {q: {} for q in queries}

        for request_key, response in responses.items():
            repo_name, query = request_key.split(":")
            response_keys = self.repos[repo_name].get(
                "response_keys", {}).get("path", [])

            extracted_data = response
            for key in response_keys:
                if isinstance(extracted_data, dict) and key in extracted_data:
                    extracted_data = extracted_data[key]
                else:
                    logger.warning(
                        f"Key '{key}' not found in response for {repo_name}")
                    extracted_data = None
                    break

            if extracted_data is not None:
                query_results[query][repo_name] = self.format_response(
                    extracted_data, repo_name)
                logger.info(
                    f"Repo: {repo_name}, Query: {query}, Retrieved {len(extracted_data)} results")

            else:
                query_results[query][repo_name] = []
                logger.error(
                    f"Could not extract data for {repo_name}, response keys path: {response_keys}")

        # Rerank results per query
        for query in queries:
            all_results = [result for repo_results in query_results[query].values()
                           for result in repo_results]
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

    def format_response(self, response: Any, repo_name: str) -> list:
        """
        Format the response from a repository.

        :param response: Response data from the repository (can be a list or a dictionary).
        :param repo_name: Name of the repository.

        :return: A list of dictionaries containing the formatted results.
        """
        geometry_field_path = self.repos[repo_name].get(
            "geometry_field_path", {}).get("path", [])
        source_url_field = self.repos[repo_name].get("source_url_field", "")
        field_mapping = self.repos[repo_name].get("field_mapping", {})

        title_fields = field_mapping.get("title", ["title"])
        description_fields = field_mapping.get("description", ["description"])
        id_fields = field_mapping.get("id", ["id"])   # ← new

        # Helper to pick first available value from nested fields
        def get_first(item, keys, default=""):
            for key in keys:
                # Split on dot notation for nested keys
                keys_split = key.split('.')
                value = item
                for sub_key in keys_split:
                    value = value.get(sub_key) if isinstance(
                        value, dict) else None
                    if value is None:
                        break
                if value:
                    return value
            return default

        # Helper to extract id
        def get_id(item, fallback_key):
            for k in id_fields:
                keys_split = k.split('.')  # Handle nested ID fields
                value = item
                for sub_key in keys_split:
                    value = value.get(sub_key) if isinstance(
                        value, dict) else None
                    if value is None:
                        break
                if value is not None:
                    return value
            # Fallback if no ID found
            return item.get("metadata", {}).get("doi") or item.get(fallback_key)

        results = []

        # If response is a dict of items
        if isinstance(response, dict) and all(not isinstance(v, list) for v in response.values()):
            for key, item in response.items():
                resolved_id = get_id(item, key)
                item["id"] = resolved_id

                results.append({
                    "id": resolved_id,
                    "source": item.get(source_url_field) or repo_name,
                    "text": f"{get_first(item, title_fields)} - {get_first(item, description_fields)}".strip(" -"),
                    "geometry_geojson": self._geometry_to_geojson(get_nested_value(item, geometry_field_path)) or "",
                    "meta": item,
                })

        # If response is a list of items
        elif isinstance(response, list):
            for item in response:
                resolved_id = get_id(item, None)
                item["id"] = resolved_id

                results.append({
                    "id": resolved_id,
                    "source": item.get(source_url_field) or repo_name,
                    "text": f"{get_first(item, title_fields)} - {get_first(item, description_fields)}".strip(" -"),
                    "geometry_geojson": self._geometry_to_geojson(get_nested_value(item, geometry_field_path)) or "",
                    "meta": item,
                })

        else:
            logger.warning(
                f"Unexpected format for {repo_name}: {type(response)}")

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
