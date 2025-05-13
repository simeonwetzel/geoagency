from smolagents import tool
from typing import List, Union, Dict, Any
from smolagents.tools import Tool
from loguru import logger


@tool 
def geocode_query(query: str) -> dict:
    """
    Can generate a BoundingBox for a query with a location entity. 
    Only use this tool if the search results you found include a geometry_geojson attribute.
    Only use this tool if the query includes a location name such as a city, country, etc.
    
    Args:
        query: The query used as input to the geocoding service.
    """ 
    import requests
    
    geocoding_endpoint = "https://photon.komoot.io/api/"
    
    params = {'q': query}

    call = requests.get(geocoding_endpoint,params=params)
    if call.status_code == 200:
        top_3_candidates = call.json().get('features')[:5]
        return f"""Here are some geocoding results for your query: {top_3_candidates}
    Use these bounding boxes to compare it with the BoundingBoxes you find in the search results `geometry_geojson` attribute.
    If you dont find any matching BoundingBox, try to think of a rough bounding box on your own."""
    else:
        return "No BoundingBox could be derived"

import json


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
        return False

    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        return False

@tool
def check_spatial_relevance(query_bbox: object, geometry_geojson: str) -> bool:
    """
    Check if a query bounding box intersects with a GeoJSON geometry.

    Args:
        query_bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
        geometry_geojson: GeoJSON string representing any geometry type

    Returns:
        Boolean indicating if the bounding box intersects with the geometry
    """
    return is_geojson_in_bbox(geometry_geojson, query_bbox)



