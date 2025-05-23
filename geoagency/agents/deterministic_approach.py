import logging
import re
import json
import time
import asyncio
from typing import Dict, List, Optional, TypedDict, Union, Any, Callable, Tuple
from dataclasses import dataclass
from functools import lru_cache

import aiohttp
import numpy as np
from loguru import logger

from geoagency.agents.retriever.retriever import RepoRetriever
from geoagency.llm_manager import LLMManager
from geoagency.agents.tools.retrieval_tools import MetadataRetrieverTool
from geoagency.agents.tools.geo_tools import geocode_query



# Initialize core components
retriever = RepoRetriever()
llm = LLMManager.get_llm()
retrieval_tool = MetadataRetrieverTool(retriever=retriever)


# ==================
# Type definitions
# ==================

@dataclass
class SearchQuery:
    keyword_query: str
    reason: str


class SearchPlan(TypedDict):
    queries: List[Dict[str, str]]
    spatial_context: str
    ready_for_search: bool
    follow_up: Optional[str]


class SearchResult(TypedDict):
    text: str
    source: str
    geometry_geojson: str
    id: Optional[str]


class GeocodingResult(TypedDict):
    geometry: Dict[str, Any]
    properties: Dict[str, Any]
    extent: Optional[List[float]]


# ======================
# Utility Functions
# ======================

@lru_cache(maxsize=128)
def parse_json_from_string(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from a string, handling cases where JSON is enclosed in backticks.
    
    Args:
        json_str: String potentially containing JSON
        
    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    # Try parsing directly
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass  # If it fails, proceed to extracting JSON from backticks

    # Extract JSON content inside triple backticks
    match = re.search(r'```json\n(.*?)\n```', json_str, re.DOTALL)
    if match:
        json_content = match.group(1)
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return None

    logger.warning("No valid JSON found in the input string.")
    return None


async def time_operation(name: str, operation: Union[Callable, Any, asyncio.coroutine]) -> Tuple[Any, float]:
    """
    Time any operation, whether it's a coroutine, callable, or direct value.

    Args:
        name: Name of the operation for timing
        operation: The operation to time (can be coroutine, callable, or value)
        
    Returns:
        Tuple of (operation result, elapsed time)
    """
    start = time.time()
    try:
        if asyncio.iscoroutine(operation):
            result = await operation
        elif callable(operation):
            result = operation()
        else:
            result = operation
        elapsed = time.time() - start
        return result, elapsed
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Operation {name} failed: {e}")
        raise


# ======================
# Geocoding Functions
# ======================

async def select_best_geocoding_match(spatial_context: str, candidates: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Use LLM to select the best match from geocoding candidates based on the spatial context.

    Args:
        spatial_context: Original spatial context string
        candidates: List of geocoding result candidates

    Returns:
        Best matching candidate with extent information
    """
    if not candidates:
        return None

    # Format candidates for LLM evaluation
    candidates_formatted = []
    for i, candidate in enumerate(candidates):
        props = candidate.get("properties", {})
        name = props.get("name", "Unknown")
        location_type = props.get("type", props.get("osm_value", "Unknown"))
        country = props.get("country", "Unknown")
        state = props.get("state", "Unknown")
        coords = candidate.get("geometry", {}).get("coordinates", [0, 0])

        # Extract or calculate extent
        extent = None
        if "extent" in props:
            extent = props["extent"]
        else:
            # Create extent from coordinates if not available
            extent = [
                coords[0] - 0.1,  # min_lon
                coords[1] - 0.1,  # min_lat
                coords[0] + 0.1,  # max_lon
                coords[1] + 0.1   # max_lat
            ]

        candidates_formatted.append({
            "id": i,
            "name": name,
            "type": location_type,
            "country": country,
            "state": state,
            "coordinates": coords,
            "extent": extent
        })

    # Prepare candidates JSON for LLM
    candidates_json = json.dumps(candidates_formatted)

    # Use LLM to select best match
    response = llm(messages=[{
        "role": "user",
        "content": f"""You are a geospatial matching expert. Select the best matching location from these candidates based on the spatial context.
        
        Spatial Context: "{spatial_context}"
        
        Candidates:
        {candidates_json}
        
        Instructions:
        1. Analyze each candidate's relevance to the spatial context
        2. Consider name similarity, administrative level, and geographic scope
        3. Provide your selection with a brief justification
        
        Return a valid JSON with this format:
        {{
            "best_match_id": <id of the best candidate>,
            "justification": "<brief explanation of your choice>"
        }}"""
    }])

    try:
        result = parse_json_from_string(response.content)
        if not result or "best_match_id" not in result:
            logger.warning("LLM did not return a valid selection")
            return candidates[0]  # Default to first candidate

        best_match_id = result["best_match_id"]
        justification = result.get("justification", "")

        logger.info(f"Selected candidate {best_match_id}: {justification}")

        # Get the original candidate and add extent
        best_candidate = candidates[best_match_id]

        # Extract or create the extent (bounding box)
        if "extent" in best_candidate.get("properties", {}):
            best_candidate["extent"] = best_candidate["properties"]["extent"]
        else:
            # If no extent is provided, create a buffer around the point
            coords = best_candidate.get("geometry", {}).get("coordinates", [0, 0])
            buffer = 0.1  # ~10km buffer
            best_candidate["extent"] = [
                coords[0] - buffer,  # min_lon
                coords[1] - buffer,  # min_lat
                coords[0] + buffer,  # max_lon
                coords[1] + buffer   # max_lat
            ]

        return best_candidate
    except Exception as e:
        logger.error(f"Error selecting best geocoding match: {e}")
        return candidates[0]  # Default to first candidate


async def geocode_spatial_context(spatial_context: str) -> Optional[Dict[str, Any]]:
    """
    Geocode the spatial context using Photon service.

    Args:
        spatial_context: String description of the spatial area

    Returns:
        Dictionary containing the best matching location with extent information
    """
    if not spatial_context:
        return None

    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://photon.komoot.io/api/?q={spatial_context}&limit=5"
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Geocoding API returned status {response.status}")
                    return None

                data = await response.json()

                if not data.get("features"):
                    logger.warning(f"No geocoding results found for '{spatial_context}'")
                    return None

                # Use LLM to select best match from candidates
                best_match = await select_best_geocoding_match(spatial_context, data["features"])
                return best_match
    except aiohttp.ClientError as e:
        logger.error(f"Client error during geocoding: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during geocoding: {e}")
        return None


# ======================
# GeoJSON Processing
# ======================

def is_geojson_in_bbox(geometry_geojson: str, bbox: List[float]) -> bool:
    """
    Check if a GeoJSON geometry intersects with a bounding box.

    Args:
        geometry_geojson: GeoJSON string representing any geometry type
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]

    Returns:
        Boolean indicating if the geometry intersects with the bounding box
    """
    if not geometry_geojson or not bbox or len(bbox) != 4:
        return False
        
    try:
        # Parse the GeoJSON
        geojson = json.loads(geometry_geojson)

        # Helper function to check if two bounding boxes intersect
        def bbox_intersects(coords_bbox, target_bbox):
            return not (
                coords_bbox[0] > target_bbox[2] or  # coords min_lon > target max_lon
                coords_bbox[2] < target_bbox[0] or  # coords max_lon < target min_lon
                coords_bbox[1] > target_bbox[3] or  # coords min_lat > target max_lat
                coords_bbox[3] < target_bbox[1]     # coords max_lat < target min_lat
            )

        # Extract geometry if it's a feature
        geometry = geojson
        if "geometry" in geojson:
            geometry = geojson["geometry"]

        if not geometry or "type" not in geometry:
            logger.warning("Invalid GeoJSON structure")
            return False

        geo_type = geometry["type"]
        coords = geometry.get("coordinates", [])

        # Calculate the bounding box for different geometry types
        if geo_type == "Point":
            # For Point, create a small bbox around it
            if not coords or len(coords) < 2:
                return False
                
            lon, lat = coords
            point_bbox = [lon - 0.0001, lat - 0.0001, lon + 0.0001, lat + 0.0001]
            return bbox_intersects(point_bbox, bbox)

        elif geo_type == "LineString" or geo_type == "MultiPoint":
            # Find min/max coordinates
            if not coords:
                return False
                
            lons = [p[0] for p in coords if isinstance(p, (list, tuple)) and len(p) >= 2]
            lats = [p[1] for p in coords if isinstance(p, (list, tuple)) and len(p) >= 2]
            
            if not lons or not lats:
                return False
                
            coords_bbox = [min(lons), min(lats), max(lons), max(lats)]
            return bbox_intersects(coords_bbox, bbox)

        elif geo_type == "Polygon" or geo_type == "MultiLineString":
            # For Polygon, check the outer ring
            if not coords or not coords[0]:
                return False

            # Flatten coordinates for MultiLineString or get outer ring for Polygon
            flat_coords = coords[0] if geo_type == "Polygon" else [
                p for line in coords for p in line if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            
            if not flat_coords:
                return False
                
            lons = [p[0] for p in flat_coords if isinstance(p, (list, tuple)) and len(p) >= 2]
            lats = [p[1] for p in flat_coords if isinstance(p, (list, tuple)) and len(p) >= 2]
            
            if not lons or not lats:
                return False
                
            coords_bbox = [min(lons), min(lats), max(lons), max(lats)]
            return bbox_intersects(coords_bbox, bbox)

        elif geo_type == "MultiPolygon":
            # Check each polygon
            for poly in coords:
                if not poly or not poly[0]:
                    continue

                # Get the outer ring
                outer_ring = poly[0]
                lons = [p[0] for p in outer_ring if isinstance(p, (list, tuple)) and len(p) >= 2]
                lats = [p[1] for p in outer_ring if isinstance(p, (list, tuple)) and len(p) >= 2]
                
                if not lons or not lats:
                    continue
                    
                poly_bbox = [min(lons), min(lats), max(lons), max(lats)]

                if bbox_intersects(poly_bbox, bbox):
                    return True

            return False
        
        # For other geometry types or if type not recognized
        logger.warning(f"Unsupported GeoJSON type: {geo_type}")
        return False

    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        logger.error(f"Error checking if geometry is in bbox: {e}")
        return False


# ======================
# Search Functions
# ======================

def generate_search_plan(user_query: str) -> SearchPlan:
    """
    Generate a structured search plan from user query using LLM.
    
    Args:
        user_query: Free-text user input
        
    Returns:
        SearchPlan with queries, spatial context, and status
    """
    try:
        response = llm(messages=[{
            "role": "user",
            "content": f"""You are an expert assistant in dataset search for Environmental and Earth System Sciences.
            Generate structured search queries from a given user input.
            Never use operators like 'AND', 'OR', 'NOT' in the queries, as they are not supported by the underlying search engine.
            
            ## Input:
            {user_query}
            
            ## Instructions:
            - Validate spatial, temporal, and thematic components. Ask for clarification if needed (If narrowing down to a location or time aspect would improve the search results).
            - Generate specific, targeted queries
            - Consider data format requirements
            
            ## Output (Valid JSON):
            {{
                "queries": [{{"keyword_query": "...", "reason": "..."}}],
                "spatial_context": "...",
                "ready_for_search": true/false,
                "follow_up": "..." # Optional (only if ready_for_search is false)
            }}"""
        }])

        plan = parse_json_from_string(response.content)
        if not plan:
            return SearchPlan(queries=[], spatial_context="", ready_for_search=False,
                             follow_up="Failed to generate search plan. Please try again.")
        return plan
    except Exception as e:
        logger.error(f"Failed to generate search plan: {e}")
        return SearchPlan(queries=[], spatial_context="", ready_for_search=False,
                         follow_up=f"Error generating search plan: {str(e)[:100]}")


async def execute_searches(search_plan: SearchPlan, bbox: Optional[List[float]] = None) -> Dict[str, List[SearchResult]]:
    """
    Execute batch searches based on search plan and filter by bounding box if provided.
    
    Args:
        search_plan: Search plan with queries
        bbox: Optional bounding box for spatial filtering
        
    Returns:
        Dictionary mapping query strings to search results
    """
    try:
        # Extract all keyword queries at once
        keyword_queries = [q["keyword_query"] for q in search_plan["queries"]]
        if not keyword_queries:
            return {}

        # Make a single batch request to retriever
        logger.info(f"Executing batch search with queries: {keyword_queries}")
        all_results = await retriever.query_multiple_repos(keyword_queries, limit=5)

        # If we have a spatial bbox, filter the results
        if bbox:
            logger.info(f"Filtering results using bounding box: {bbox}")
            filtered_results = {
                query: [res for res in results if not res.get('geometry_geojson') or 
                        is_geojson_in_bbox(res['geometry_geojson'], bbox)]
                for query, results in all_results.items()
            }
            
            # Only use filtered results if we actually have some results left
            if any(filtered_results.values()):
                return filtered_results
                
            logger.warning("Spatial filtering removed all results. Using unfiltered results.")
            
        return all_results

    except Exception as e:
        logger.error(f"Batch search operation failed: {e}")
        return {}


def format_search_results(
    search_plan: SearchPlan, 
    search_results: Dict[str, List[SearchResult]],
    geocoding_result: Optional[Dict[str, Any]] = None,
    bbox: Optional[List[float]] = None
) -> Tuple[str, str]:
    """
    Format search results into human-readable strings.
    
    Args:
        search_plan: Original search plan
        search_results: Dictionary of search results
        geocoding_result: Optional geocoding information
        bbox: Optional bounding box
        
    Returns:
        Tuple of (queries text, results text)
    """
    # Format queries
    queries_format = []
    for q in search_plan["queries"]:
        query = q.get('keyword_query')
        if query and query in search_results:
            reason = q.get('reason', 'No reason specified')
            queries_format.append(f"- **{query}**: {reason}")
    
    queries_text = "\n".join(queries_format) if queries_format else "No queries executed."

    # Include spatial context information in the results if available
    spatial_context = search_plan.get("spatial_context", "")
    spatial_info = ""
    if geocoding_result:
        props = geocoding_result.get('properties', {})
        spatial_info = (f"\nSpatial Context: {props.get('name', spatial_context)}"
                        f"\nRegion: {props.get('state', '')} {props.get('country', '')}"
                        f"\nBounding Box: {bbox}")

    # Format results
    results_format = []
    for query_results in search_results.values():
        for i, result in enumerate(query_results[:3]):  # Limit to top 3 per query
            text = result.get('text', 'No description')[:100]
            keywords = result.get('keywords', 'No keywords')
            source = result.get('source', '#')
            result_id = result.get('id', 'No ID')
            geo = result.get('geometry_geojson', 'Not specified')
            
            results_format.append(
                f"\n-- Text: [{text}](Source: {source})\nKeywords: {keywords}\n"
                f"  ID: {result_id}\n"
                f"  Region: {geo}"
            )
    
    results_text = "\n".join(results_format) if results_format else "No relevant results found."

    return queries_text + spatial_info, results_text


def evaluate_results(
    query: str,
    search_results: Dict[str, List[SearchResult]],
    queries_text: str,
    result_text: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Use LLM to provide a compact textual evaluation of the search results.
    
    Args:
        query: Original user query
        search_plan: Search plan with queries
        search_results: Raw search results dictionary
        queries_text: Formatted text of search queries used
        
    Returns:
        Tuple of evaluation text and list of top reranked results
    """
    try:
        # Extract all unique result IDs for later mapping
        all_result_ids = set()
        id_to_result_map = {}
        for query_results in search_results.values():
            for result in query_results:
                result_id = result.get('id')
                if result_id:
                    all_result_ids.add(result_id)
                    id_to_result_map[result_id] = result
        
        # No results to evaluate
        if not id_to_result_map:
            return "No results to evaluate.", []
            
        response = llm(messages=[{
            "role": "user",
            "content": f"""Evaluate these dataset search results:
            Query: {query}
            
            Search Queries Used:
            {queries_text}

            Available Results (top candidates):
            {result_text}

            ## Instructions
            1. Provide a compact evaluation (max 150 words) highlighting:
            - Which results are most relevant to the query
            - Focus on spatial relevance if applicable
            - Key insights or limitations in the results
            
            2. Re-rank the top results by relevance to the query
            
            ## Output format (JSON):
            {{
                "evaluation": "Your concise evaluation here",
                "top_10": ["id1", "id2", "id3", ...]  // List of result IDs in ranked order
            }}
            """
            }])
        
        # Parse the response
        result = parse_json_from_string(response.content)
        if not result or "evaluation" not in result:
            logger.warning("LLM did not return a valid evaluation")
            return "Could not generate a proper evaluation of the results.", []
        
        # Convert top_10 IDs to actual result objects
        top_reranked = []
        if "top_10" in result and isinstance(result["top_10"], list):
            for result_id in result["top_10"]:
                if result_id in id_to_result_map:
                    top_reranked.append(id_to_result_map[result_id])
                    
        return result["evaluation"], top_reranked 
    
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return f"Error evaluating results: {str(e)[:100]}", []


# ======================
# Main Function
# ======================

async def call_agent(query: str, use_follow_ups: str) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced workflow for conversational dataset search with improved robustness,
    caching, and parallel processing capabilities.
    
    Args:
        query: User's natural language query
        
    Returns:
        Tuple of (response text, search results dictionary)
    """
    
    logger.info("Using deterministic approach for dataset search.")
    logger.info(f"Using follow-ups: {use_follow_ups}")      
    timings = {}
    try:
        # Step 1: Generate and validate search plan
        search_plan, timings['query_generation'] = await time_operation(
            'query_generation',
            lambda: generate_search_plan(query)
        )
        if search_plan and use_follow_ups == "no":
            search_plan['ready_for_search'] = True  
        
        if not search_plan or not search_plan.get("ready_for_search"):
            follow_up = (search_plan or {}).get("follow_up", "Additional information needed.")
            return follow_up, {}
        
        logger.info(f"Search plan generated: {search_plan}")
        
        logger.info(f"Checking if search plan also includes original query: {query}")
        if query not in [q["keyword_query"] for q in search_plan["queries"]]:
            search_plan["queries"].append({
                "keyword_query": query,
                "reason": "Original query"
            })
        
        # Step 2: Process spatial context and geocode it
        spatial_context = search_plan.get('spatial_context', '')
        geocoding_result = None
        bbox = None

        if spatial_context:
            geocoding_result, timings['geocoding'] = await time_operation(
                'geocoding',
                geocode_spatial_context(spatial_context)
            )

            if geocoding_result:
                logger.info(f"Geocoded '{spatial_context}' to {geocoding_result.get('properties', {}).get('name')}")
                bbox = geocoding_result.get('extent')
                logger.info(f"Using bounding box: {bbox}")

        # Step 3: Execute searches using batch processing
        search_results, timings['retrieval'] = await time_operation(
            'retrieval',
            execute_searches(search_plan, bbox)
        )

        if not search_results:
            return "No relevant datasets found.", {}

        # Step 4: Process and format results
        (queries_text, results_text), timings['formatting'] = await time_operation(
            'formatting',
            lambda: format_search_results(search_plan, search_results, geocoding_result, bbox)
        )

        # Step 5: Evaluate results
        (evaluation_text, reranked_results), timings["evaluation"] = await time_operation(
            'evaluation',
            lambda: evaluate_results(query, search_results, queries_text, results_text)
        )
                
        # Format timing information
        logger.warning(f"Timing information: {timings}")
        timing_summary = "\n".join(
            f"{operation}: {duration:.2f}s"
            for operation, duration in timings.items()
        )
        
        final_search_results = {
            'search_results': search_results,
            're-ranked_results': reranked_results
        }

        return f"{evaluation_text}\n\nPerformance Metrics:\n{timing_summary}", final_search_results
        
    except Exception as e:
        logger.error(f"Error in call_agent: {e}")
        # Include any timing information collected before the error
        timing_summary = "\n".join(
            f"{operation}: {duration:.2f}s"
            for operation, duration in timings.items()
        )
        return f"Error processing search: {str(e)}\n\nPartial Performance Metrics:\n{timing_summary}", {}