import asyncio
import aiohttp
import requests
from loguru import logger
from flashrank import Ranker, RerankRequest
import time
import shapely
from ...config.config import CONFIG
from typing import Any, Dict, List, Optional, Union
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
                geometry_coordinate_order=repo_config.get("geometry_coordinate_order", "lat_lon"), 
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
                 geometry_coordinate_order: str = None,
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
            "geometry_coordinate_order": geometry_coordinate_order or "lat_lon",
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
            
            # Log error if status code is not 200
            if response.status_code != 200:
                logger.error(f"Health check for repository '{repo_name}' failed with status code {response.status_code}: {response.text}")
                return {"status": "unhealthy", "error": f"Status code {response.status_code}", "response": response.text}
            
            response.raise_for_status()
            return {"status": "healthy"}
        except Exception as e:
            logger.error(f"Health check for repository '{repo_name}' failed with exception: {str(e)}")
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
                async with session.get(url, params=constructed, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Request to repository '{repo_name}' failed with status code {response.status}: {error_text}")
                        return {"error": f"Status code {response.status} for {repo_name}", "response": error_text}
                    response.raise_for_status()
                    return await response.json()
            
            # POST requests
            if query_format == "json_template":
                body = constructed.get("body", {})
                async with session.post(url, json=body, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Request to repository '{repo_name}' failed with status code {response.status}: {error_text}")
                        return {"error": f"Status code {response.status} for {repo_name}", "response": error_text}
                    response.raise_for_status()
                    return await response.json()
            else:
                async with session.post(url, json=constructed, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Request to repository '{repo_name}' failed with status code {response.status}: {error_text}")
                        return {"error": f"Status code {response.status} for {repo_name}", "response": error_text}
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as ce:
            logger.error(f"Network error when requesting repository '{repo_name}': {str(ce)}")
            return {"error": f"Network error for {repo_name}: {str(ce)}"}
        except ValueError as ve:
            logger.error(f"JSON parsing error for repository '{repo_name}': {str(ve)}")
            return {"error": f"JSON parsing error for {repo_name}: {str(ve)}"}
        except Exception as e:
            logger.error(f"Unknown error when requesting repository '{repo_name}': {str(e)}")
            return {"error": f"Unknown error for {repo_name}: {str(e)}"}

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
            
            # Process and log any exceptions that occurred during gathering results
            processed_results = {}
            for (key, _), result in zip(tasks, results):
                repo_name = key.split(":")[0]
                if isinstance(result, Exception):
                    logger.error(f"Exception occurred during request to repository '{repo_name}': {str(result)}")
                    processed_results[key] = {"error": f"Exception: {str(result)}"}
                else:
                    processed_results[key] = result
                    
                    # Log if the result contains an error
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"Error in response from repository '{repo_name}': {result['error']}")
                        
            return processed_results

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
            
            # Check if the response contains an error
            if isinstance(response, dict) and "error" in response:
                logger.error(f"Error in response from repository '{repo_name}' for query '{query}': {response['error']}")
                query_results[query][repo_name] = []
                continue
                
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

    def _geometry_to_geojson(self, geometry: Any, repo_name: str) -> Union[dict, str]:
        """
        Convert geometry to GeoJSON format, handling different coordinate orders.
        
        :param geometry: Input geometry (WKT string, dict, etc.)
        :param repo_name: Repository name to determine coordinate order
        :return: GeoJSON representation of the geometry
        """
        if not geometry:
            return ""
            
        # Get coordinate order from repository config
        coordinate_order = self.repos[repo_name].get("geometry_coordinate_order", "lat_lon")
        
        try:
            if isinstance(geometry, str):
                # Handle WKT string
                geometry_obj = shapely.from_wkt(geometry=geometry)
                geojson = shapely.to_geojson(geometry=geometry_obj)
                return self._normalize_coordinate_order(geojson, coordinate_order)
                
            elif isinstance(geometry, list):
                geometry = geometry[0]
                
            if isinstance(geometry, dict):
                if 'min' in geometry and 'max' in geometry:
                    # Handle bounding box format
                    polygon = shapely.Polygon([
                        # Bottom-left
                        (geometry["min"]["x"], geometry["min"]["y"]),
                        # Bottom-right
                        (geometry["max"]["x"], geometry["min"]["y"]),
                        # Top-right
                        (geometry["max"]["x"], geometry["max"]["y"]),
                        (geometry["min"]["x"], geometry["max"]["y"]),  # Top-left
                        # Close the polygon
                        (geometry["min"]["x"], geometry["min"]["y"])
                    ])
                    geojson = shapely.to_geojson(geometry=polygon)
                    return self._normalize_coordinate_order(geojson, coordinate_order)
                elif 'type' in geometry and 'coordinates' in geometry:
                    # Already in GeoJSON format
                    return self._normalize_coordinate_order(geometry, coordinate_order)
                    
        except Exception as e:
            logger.error(f"Error converting geometry to GeoJSON for repo {repo_name}: {str(e)}")
            
        return ""

    def _normalize_coordinate_order(self, geojson_dict: Union[dict, str], source_order: str) -> dict:
        """
        Normalize coordinate order in GeoJSON to always use [lat, lon] order.
        
        :param geojson_dict: GeoJSON dictionary or string
        :param source_order: Source coordinate order ('lon_lat' or 'lat_lon')
        :return: Normalized GeoJSON dictionary
        """
        # Convert string to dict if needed
        if isinstance(geojson_dict, str):
            try:
                geojson_dict = json.loads(geojson_dict)
            except json.JSONDecodeError:
                logger.error("Failed to parse GeoJSON string")
                return geojson_dict
                
        # If coordinates are already in [lon, lat] order, return as is
        if source_order == "lat_lon":
            return geojson_dict
            
        # Handle Feature object
        if 'type' in geojson_dict and geojson_dict['type'] == 'Feature':
            geometry = geojson_dict.get('geometry', {})
            if not geometry:
                return geojson_dict
            geojson_dict['geometry'] = self._swap_coordinates(geometry)
            return geojson_dict
            
        # Handle direct geometry object
        if 'type' in geojson_dict and 'coordinates' in geojson_dict:
            return self._swap_coordinates(geojson_dict)
            
        return geojson_dict

    def _swap_coordinates(self, geometry: dict) -> dict:
        """
        Swap coordinates in a GeoJSON geometry from [lat, lon] to [lon, lat] or vice versa.
        
        :param geometry: GeoJSON geometry object
        :return: Geometry with swapped coordinates
        """
        geom_type = geometry.get('type', '')
        coordinates = geometry.get('coordinates', [])
        
        if geom_type == 'Point':
            if len(coordinates) >= 2:
                geometry['coordinates'] = [coordinates[1], coordinates[0]]
                
        elif geom_type == 'LineString' or geom_type == 'MultiPoint':
            geometry['coordinates'] = [[coord[1], coord[0]] for coord in coordinates]
            
        elif geom_type == 'Polygon' or geom_type == 'MultiLineString':
            geometry['coordinates'] = [[[coord[1], coord[0]] for coord in ring] for ring in coordinates]
            
        elif geom_type == 'MultiPolygon':
            geometry['coordinates'] = [[[[coord[1], coord[0]] for coord in ring] for ring in polygon] for polygon in coordinates]
            
        return geometry

    def validate_geometry(self, geojson_str: Union[str, dict]) -> bool:
        """
        Validate if a GeoJSON string is correctly formatted.
        
        :param geojson_str: GeoJSON string or dictionary
        :return: True if valid, False otherwise
        """
        if not geojson_str:
            return False
            
        try:
            if isinstance(geojson_str, str):
                geojson_dict = json.loads(geojson_str)
            else:
                geojson_dict = geojson_str
                
            # Check if it's a valid GeoJSON
            if 'type' not in geojson_dict:
                return False
                
            # Basic check for coordinates (could be expanded)
            if 'coordinates' in geojson_dict:
                coords = geojson_dict['coordinates']
                if not coords:
                    return False
            elif 'geometry' in geojson_dict and 'coordinates' in geojson_dict['geometry']:
                coords = geojson_dict['geometry']['coordinates']
                if not coords:
                    return False
            else:
                return False
                
            return True
            
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"GeoJSON validation error: {str(e)}")
            return False
        
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
                
                # Get geometry and convert to GeoJSON
                geom = get_nested_value(item, geometry_field_path)
                geojson = self._geometry_to_geojson(geom, repo_name)
                
                # Validate the geometry
                if geojson:
                    if self.validate_geometry(geojson):
                        geojson = json.dumps(geojson)  # Ensure it's a string
                    else:
                        logger.warning(f"Invalid geometry detected for item {resolved_id} in repo {repo_name}")
                        geojson = ""
                    
                

                results.append({
                    "id": resolved_id,
                    "source": item.get(source_url_field) or repo_name,
                    "text": f"{get_first(item, title_fields)} - {get_first(item, description_fields)}".strip(" -"),
                    "geometry_geojson": geojson,
                    "meta": item,
                })

        # If response is a list of items
        elif isinstance(response, list):
            for item in response:
                resolved_id = get_id(item, None)
                item["id"] = resolved_id
                
                # Get geometry and convert to GeoJSON
                geom = get_nested_value(item, geometry_field_path)
                geojson = self._geometry_to_geojson(geom, repo_name)
                
                # Validate the geometry
                if geojson:
                    if self.validate_geometry(geojson):
                        geojson = json.dumps(geojson)  # Ensure it's a string
                    else:
                        logger.warning(f"Invalid geometry detected for item {resolved_id} in repo {repo_name}")
                        geojson = ""

                results.append({
                    "id": resolved_id,
                    "source": item.get(source_url_field) or repo_name,
                    "text": f"{get_first(item, title_fields)} - {get_first(item, description_fields)}".strip(" -"),
                    "geometry_geojson": geojson,
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
                
            # Check if the response contains an error
            if isinstance(response, dict) and "error" in response:
                logger.error(f"Error in response from repository '{repo_name}' for query '{query}': {response['error']}")
                continue

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