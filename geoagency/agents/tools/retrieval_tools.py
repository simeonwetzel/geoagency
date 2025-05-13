from smolagents import Tool, tool
from typing import List, Dict, Any
from geoagency.agents.tools.geo_tools import is_geojson_in_bbox
from loguru import logger
import re

class DecomposeQueryTool(Tool):
    name = "query_decomposer"
    description = """This tool decomposes a query into a list of 5 queries to cover all aspects of the request"""
    inputs = {
        "query": {
            "type": "string",
            "description": "User request to decompose"
        }
    }
    output_type = "object"

    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    def forward(self, query: str) -> List[str]:
        import re
        prompt = f"""Generate a list of 5 keyword queries decomposed from this request: {query}.
        Only output a python list of strings (no further text, or description)
        List:"""
        generation = self.llm(messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }])
        # Regex to extract strings inside double quotes within brackets
        matches = re.findall(r'\["(.*?)"\]|\s*"([^"]+)"', generation.content)

        # Flatten matches and filter out empty entries
        result = [m[0] or m[1] for m in matches]

        logger.info(f"Query decomposer: {result}")

        return result

class MetadataRetrieverTool(Tool):
    name = "retrieve_metadata"
    description = """Uses the RepoRetriever class to retrieve the metadata records. 
    Always use this tool to get the metadata records from the repositories.
    
    This tool returns a formatted string as summary of the metadata records retrieved. 
    
    This tool requires a list of **at least** 3-5 items like ["<query1>", "<query2>", ... ,"<query5>"], which are the queries to be searched.
    
    If the user query includes a location reference (e.g. city, country, region or explicit bounding box), think of a reasonable bounding box from it (not []).
    
    If you pass a bounding box, take care that the bounding box is a list of floats [min_lat, min_lon, max_lat, max_lon].
        
    This tool will retrieve results for each query, store them, and report how many unique results were found.
    """
    inputs = {
        "queries": {
            "type": "object",
            "description": "A list of queries decomposed from the user request.",

        },
        "bbox": {
            "type": "object",
            "description": "A bounding box for the spatial scope (optional). Format [min_lat, min_lon, max_lat, max_lon].",
            "nullable": True,
        }
    }
    output_type = "string"

    def __init__(self, retriever, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.agent_instance = None

    def set_agent_instance(self, agent):
        """Store reference to the agent instance."""
        self.agent_instance = agent

    def forward(self, queries: List[str], bbox: List[float] = [-90.0, -180.0, 90.0, 180.0]) -> str:
        import asyncio

        results = asyncio.run(
            self.retriever.query_multiple_repos(queries=queries, limit=10))

        # spatial filtering:
        if bbox is not None:
            filtered_results = {}
            for query, items in results.items():
                filtered_items = []
                for item in items:
                    if item.get('geometry_geojson') and is_geojson_in_bbox(item['geometry_geojson'], bbox):
                        filtered_items.append(item)
                    elif not item.get('geometry_geojson'):
                        filtered_items.append(item)
                if len(filtered_items) < len(items):
                    logger.info(f"Spatial filtering resulted in a subset of {len(filtered_items)} out of {len(items)} items for query '{query}' based on bounding box: {bbox}.")
                else:
                    logger.info("Spatial filter passed. No results a filtered out.")
                    
                filtered_results[query] = filtered_items

        else:
            filtered_results = results

        # Calculate total unique results
        all_items = [item for sublist in filtered_results.values()
                     for item in sublist]
        unique_items = list({v['id']: v for v in all_items}.values())
        
        print(f"Collected {len(unique_items)} results")

        self.agent_instance.last_full_results = unique_items

        return f"Found {len(unique_items)} records. Saved it to the agent instance. Use the re-ranking tool (which can access the records from the agent instance)."
        
@tool
def understand_query(thematic_scope: str,
                    spatial_scope: str,
                    temporal_scope: str,
                    bbox: object) -> dict:
    """Use this to understand the query and extract the thematic, spatial, and temporal scopes.
    Note that not all queries will have all three scopes.
    BBOX is a list of floats [min_lat, min_lon, max_lat, max_lon].

    Args:
        thematic_scope: Thematic scope of the query
        spatial_scope: Spatial scope of the query
        temporal_scope: Temporal scope of the query
        bbox: Bounding box for the spatial scope (optional)
    """

    query_criteria = {
        "thematic_scope": thematic_scope,
        "spatial_scope": spatial_scope,
        "temporal_scope": temporal_scope,
        "bbox": bbox
    }

    # Store the query criteria in the agent instance if possible
    caller = getattr(understand_query, 'caller', None)
    if hasattr(caller, 'agent_instance') and caller.agent_instance is not None:
        caller.agent_instance.query_criteria = query_criteria

    return query_criteria

def _format_results_for_llm(results, include_geometry=False):
        """
        Convert a list of search results to a markdown format that's easily consumable by an LLM.

        Args:
            results: List of result dictionaries
            include_geometry: Whether to include geometry data (defaults to False to avoid Ellipsis issues)

        Returns:
            String with formatted markdown
        """
        markdown = "# Search Results\n\n"

        for i, result in enumerate(results, 1):
            # Create header with result number and ID (truncated if too long)
            id_display = result.get('id', 'unknown')
            if len(id_display) > 15:
                id_display = id_display[:12] + "..."

            markdown += f"## Result {i}: {id_display}\n\n"

            # Add source link if available
            if 'source' in result and result['source']:
                markdown += f"**Source:** {result['source']}\n\n"

            # Add score if available
            if 'score' in result and result['score'] is not None:
                markdown += f"**Relevance Score:** {result['score']:.2f}\n\n"

            # Add text content if available
            if 'text' in result and result['text']:
                # Truncate text if it's extremely long
                text = result['text']
                if len(text) > 500:
                    text = text[:497] + "..."
                markdown += f"**Content:**\n\n{text}\n\n"

            # Add geometry if requested and available
            if include_geometry and 'geometry_geojson' in result and result['geometry_geojson']:
                markdown += f"**Geometry:** {result['geometry_geojson']}\n\n"

            # Add separator between results
            markdown += "---\n\n"

        return markdown
    
class ResultReRankingTool(Tool):
    name = "re_ranker"
    description = """This tool re-ranks metadata based on spatio-temporal and thematic relevance and outputs. Only use once per search turn.
    This tool only needs a query as argument. The metadata is provided via the agent instance that this tool can access"""
    inputs = {
        "query": {
            "type": "string",
            "description": "User request to decompose"
        }
    }
    output_type = "object"

    def __init__(self, llm, batch_size: int = 20, max_retries: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.agent_instance = None

    def _parse_enumerated_scores(self, text: str, expected: int) -> List[float]:
        """
        Parse lines like `SCORE 1: 0.23` and return floats in order.
        """
        pattern = r"SCORE\s*(\d+):\s*(-?\d+(?:\.\d+)?)"
        matches = re.findall(pattern, text)
        scores = [None] * expected
        for idx_str, val_str in matches:
            idx = int(idx_str) - 1
            if 0 <= idx < expected:
                try:
                    scores[idx] = float(val_str)
                except ValueError:
                    pass
        # Ensure all filled
        if any(s is None for s in scores):
            return []
        return scores

    def forward(self, query: str) -> List[Dict[str, Any]]:
        results = self.agent_instance.last_full_results
        print(f"The agent provided {len(results)} items to re-rank")
        if not results:
            logger.warning("No results to re-rank")
            return []

        all_scores: List[float] = []
        for batch_start in range(0, len(results), self.batch_size):
            batch = results[batch_start:batch_start + self.batch_size]
            
            n = len(batch)
            batch_md = _format_results_for_llm(batch)

            prompt_header = (
                f"CRITICAL: You must output EXACTLY {n} scores, one per line, in the format 'SCORE 1: 0.00',... 'SCORE {n}: 1.00', and nothing else."
            )
            prompt_body = (
                f"User Query: \"{query}\"\n\n"
                f"Here are {n} search results to score by thematic, spatial, and temporal relevance:\n{batch_md}\n"
                f"Provide scores between 0.0 and 1.0."  
            )

            for attempt in range(1, self.max_retries + 1):
                logger.debug(f"Batch {batch_start//self.batch_size+1}: attempt {attempt}")
                response = self.llm([
                    {"role": "system", "content": [{"type": "text", "text": prompt_header}]},
                    {"role": "user", "content": [{"type": "text", "text": prompt_body}]}
                ])
                logger.debug(f"LLM raw response:\n{response}")

                # Parse scores
                response_text = response.content
                scores = self._parse_enumerated_scores(response_text, n)
                if len(scores) == n:
                    logger.info(f"Parsed {n} scores successfully")
                    break
                else:
                    logger.warning(
                        f"Parsed {len(scores)} scores, expected {n}. Retrying..."
                    )
            else:
                logger.error(
                    f"Failed to get {n} scores after {self.max_retries} attempts. Using neutral 0.5s."
                )
                scores = [0.5] * n

            # Clip scores to [0.0, 1.0]
            normalized = [max(0.0, min(1.0, s)) for s in scores]
            all_scores.extend(normalized)

        # Ensure top-level count correctness
        if len(all_scores) != len(results):
            logger.error(
                f"Total scored count {len(all_scores)} != results count {len(results)}. Adjusting."
            )
            if len(all_scores) > len(results):
                all_scores = all_scores[:len(results)]
            else:
                all_scores.extend([0.5] * (len(results) - len(all_scores)))

        # Re-rank
        paired = list(zip(all_scores, results))
        ranked = [res for _, res in sorted(paired, key=lambda x: x[0], reverse=True)]
        self.agent_instance.ranked_results = ranked
        return _format_results_for_llm(ranked[:10])
